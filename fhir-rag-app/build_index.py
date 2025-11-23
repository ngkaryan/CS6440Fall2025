# build_index.py
# This runs ONLY during Render build (once per deploy)
# It generates Synthea data → builds deduplicated FAISS index → saves files for fast startup

import os
import json
import glob
import pickle
import numpy as np
import faiss
import logging
import time
from tqdm import tqdm
import google.generativeai as genai
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_EMBED_MODEL = "models/embedding-001"
BATCH_SIZE = 50
MAX_CHUNK = 1200
OVERLAP = 200
INDEX_FILE = "faiss_index.bin"
EMB_FILE = "embeddings.npy"
TEXT_FILE = "index_texts.pkl"


# ======================= YOUR ORIGINAL HELPERS =======================
def chunk_text(text: str) -> List[str]:
    if len(text) <= MAX_CHUNK:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + MAX_CHUNK
        chunks.append(text[start:end])
        start = end - OVERLAP
        if start >= len(text):
            break
    return chunks


def fhir_to_text(resource: Dict[str, Any]) -> str:
    rt = resource.get("resourceType", "Unknown")
    rid = resource.get("id", "")
    # Your original beautiful logic (keep exactly as you wrote it)
    if rt == "Patient":
        try:
            from pydantic import BaseModel

            class HumanName(BaseModel):
                given: list[str] = []
                family: str = ""
                use: str = ""

                def __str__(self):
                    return f"{', '.join(self.given)} {self.family}".strip()

            class Patient(BaseModel):
                resourceType: str
                id: str
                name: list[HumanName] = []
                birthDate: str = ""
                gender: str = ""

            p = Patient(**resource)
            name = next(
                (str(n) for n in p.name if n.use in ("official", "usual")),
                str(p.name[0]) if p.name else "Unknown",
            )
            return f"Patient | ID:{rid} | Name:{name} | Gender:{p.gender} | DOB:{p.birthDate}"
        except:
            pass

    extra = []
    for k, v in resource.items():
        if k in {"text", "contained", "extension", "meta"}:
            continue
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, str) and len(v) > 200:
            v = v[:197] + "..."
        extra.append(f"{k}:{v}")
        if len(extra) >= 4:
            break
    return f"{rt} | ID:{rid} " + " | ".join(extra)


def embed_batch(batch: List[str]):
    time.sleep(0.35)  # stay under Gemini rate limits
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=batch,
        task_type="retrieval_document",
    )
    return result["embedding"]


# ======================= YOUR ORIGINAL build_faiss_index_dedup_v3 =======================
def build_faiss_index_dedup_v3():
    print("\n" + "=" * 60)
    print("BUILDING DEDUPLICATED INDEX")
    print("=" * 60)

    seen_ids = set()
    unique_resources = []
    stats = {}

    for path in sorted(glob.glob("synthea_output/fhir/*.json")):
        fname = os.path.basename(path)
        try:
            with open(path) as f:
                bundle = json.load(f)
            if bundle.get("resourceType") != "Bundle":
                continue
            entries = bundle.get("entry", [])
            print(f" {fname:25} → {len(entries):4} entries", end="")
            new = 0
            for entry in entries:
                res = entry.get("resource", {})
                res_id = res.get("id")
                if not res_id and "fullUrl" in entry:
                    res_id = entry["fullUrl"].split("/")[-1]
                if res_id and res_id not in seen_ids:
                    seen_ids.add(res_id)
                    unique_resources.append(res)
                    rt = res.get("resourceType", "Unknown")
                    stats[rt] = stats.get(rt, 0) + 1
                    new += 1
            print(f" → +{new} new")
        except Exception as e:
            print(f"\n FAILED {fname}: {e}")

    print(f"\n→ TOTAL UNIQUE: {len(unique_resources)}")
    for t, c in sorted(stats.items(), key=lambda x: -x[1]):
        print(f" {t:12}: {c}")

    print("\n2. Flattening & chunking...")
    all_texts = []
    for r in tqdm(unique_resources, desc="Flattening", unit="res"):
        txt = fhir_to_text(r)
        if txt.strip():
            all_texts.extend(chunk_text(txt))

    # Safety deduplication
    all_texts = list(dict.fromkeys(all_texts))
    print(f"→ DEDUPLICATED CHUNKS: {len(all_texts)}")

    print("\n3. Embedding with Gemini...")
    embeddings = []
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Batches"):
        batch = all_texts[i : i + BATCH_SIZE]
        embeddings.extend(embed_batch(batch))

    if not embeddings:
        raise RuntimeError("No embeddings generated")

    print("\n4. Building FAISS index...")
    arr = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL22(arr.shape[1])
    index.add(arr)

    # Save everything
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, arr)
    with open(TEXT_FILE, "wb") as f:
        pickle.dump(all_texts, f)

    print(f"Index built and saved: {index.ntotal} vectors, dim={arr.shape[1]}")
    return index, all_texts


# ======================= RUN IT =======================
if __name__ == "__main__":
    build_faiss_index_dedup_v3()
