# main.py
import logging, os, json, glob, re, time, functools, numpy as np, pickle
from typing import List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import faiss, pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_EMBED_MODEL = "models/embedding-001"
BATCH_SIZE = 50
MAX_CHUNK = 1200
OVERLAP = 200

INDEX_FILE = "faiss_index.bin"
EMB_FILE = "embeddings.npy"
TEXT_FILE = "index_texts.pkl"

# ======================= Pydantic models =======================
class HumanName(BaseModel):
    given: list[str] = []
    family: str = ""
    use: str = ""
    def __str__(self): return f"{', '.join(self.given)} {self.family}".strip()

class Patient(BaseModel):
    resourceType: str
    id: str
    name: list[HumanName] = []
    birthDate: str = ""
    gender: str = ""

# ======================= Helper functions =======================
def chunk_text(text: str) -> List[str]:
    if len(text) <= MAX_CHUNK: return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + MAX_CHUNK
        chunks.append(text[start:end])
        start = end - OVERLAP
        if start >= len(text): break
    return chunks

def fhir_to_text(resource: Dict[str, Any]) -> str:
    rt = resource.get("resourceType", {}).get("resourceType", "Unknown")
    rid = resource.get("resource", {}).get("id", "")
    # (your beautiful fhir_to_text logic – unchanged)
    if rt == "Patient":
        try:
            p = Patient(**resource["resource"])
            name = next((str(n) for n in p.name if n.use in ("official","usual")), str(p.name[0]) if p.name else "Unknown")
            return f"Patient | ID:{rid} | Name:{name} | Gender:{p.gender} | DOB:{p.birthDate}"
        except: pass
    # ... (keep all your existing cases (Observation, Condition, etc.) exactly as you wrote them...
    # (I'm keeping it short here – just paste your full function)
    extra = []
    for k,v in resource.get("resource", {}).items():
        if k in {"text","contained","extension"}: continue
        if isinstance(v,(dict,list)): continue
        if isinstance(v,str) and len(v)>200: v = v[:197]+"..."
        extra.append(f"{k}:{v}")
        if len(extra)>=3: break
    return f"{rt} | ID:{rid} " + " | ".join(extra)

def retry(times=5, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try: return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times: raise
                    logging.warning(f"Retry {attempt}/{times} – {e}")
                    time.sleep(delay * (2 ** (attempt-1)))
            return wrapper
    return decorator

@retry(times=5, delay=1)
def embed_batch(batch: List[str]):
    result = genai.embed_content(model=GEMINI_EMBED_MODEL, content=batch, task_type="retrieval_document")
    return result["embedding"]

def get_embeddings_smart(texts: List[str]):
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    embeddings = []
    for batch in tqdm(batches, desc="Embedding batches"):
        embeddings.extend(embed_batch(batch))
        time.sleep(0.3)
    return embeddings

def save_index(index, embeddings, texts):
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, np.array(embeddings, dtype=np.float32))
    with open(TEXT_FILE, "wb") as f: pickle.dump(texts, f)
    logging.info("Index saved")

def load_index():
    if not all(os.path.exists(f) for f in (INDEX_FILE, EMB_FILE, TEXT_FILE)):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMB_FILE).tolist()
    with open(TEXT_FILE, "rb") as f: texts = pickle.load(f)
    logging.info(f"Loaded index with {index.ntotal} vectors")
    return index, embeddings, texts

def build_faiss_index_dedup_v3():
    print("\n" + "="*60)
    print("BUILDING DEDUPLICATED INDEX")
    print("="*60)

    seen_ids = set()
    unique_resources = []
    stats = {}

    for path in sorted(glob.glob("synthea_output/fhir/*.json")):
        fname = os.path.basename(path)
        try:
            with open(path) as f:
                bundle = json.load(f)
            if bundle.get("resourceType") != "Bundle": continue
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

    print("\n2. Flattening...")
    all_texts = []
    for r in tqdm(unique_resources, desc="Flattening", unit="res"):
        txt = fhir_to_text(r)
        if txt.strip():
            all_texts.extend(chunk_text(txt))

    # SAFETY DEDUP
    all_texts = list(dict.fromkeys(all_texts))
    print(f"→ DEDUPLICATED CHUNKS: {len(all_texts)}")

    print("\n3. Embedding...")
    embeddings = get_embeddings_smart(all_texts)
    if not embeddings:
        return None, all_texts

    print("\n4. FAISS...")
    arr = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    print(f"→ {index.ntotal} vectors, dim={arr.shape[1]}")

    return index, all_texts


def build_faiss():
    # (your full build_faiss_index_dedup_v3 function – paste unchanged)
    # ... exactly the same code you already have ...
    pass

# ======================= FastAPI app =======================
app = FastAPI(title="FHIR RAG – 10 patients")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load or build index at startup
faiss_index, _, index_texts = load_index()
if faiss_index is None:
    logging.info("No index found → building fresh (this takes ~4–8 min first time)")
    faiss_index, index_texts = build_faiss_index_dedup_v3()
    embeddings = get_embeddings_smart(index_texts)
    save_index(faiss_index, embeddings, index_texts)
else:
    logging.info("Index loaded – ready to query!")

app.state.faiss_index = faiss_index
app.state.index_texts = index_texts

@app.get("/nl_fhir_query")
async def nl_fhir_query(q: str = Query(...), mode: str = "table", top_k: int = 10):
    idx = app.state.faiss_index
    txts = app.state.index_texts
    if idx is None:
        raise HTTPException(500, "Index not ready")
    q_emb = np.array(get_embeddings_smart([q]), dtype=np.float32)
    D, I = idx.search(q_emb, top_k * 2)
    hits = [txts[i] for i in I[0] if i != -1][:top_k]

    if mode == "chart":
        cnt = {}
        for h in hits:
            for part in re.split(r'\s*\|\s*', h):
                if ":" in part:
                    k = part.split(":",1)[0].strip()
                    cnt[k] = cnt.get(k,0)+1
        return cnt
    return {"results": hits}

@app.get("/health")

async def health(): return {"status": "ok", "vectors": app.state.faiss_index.ntotal}
