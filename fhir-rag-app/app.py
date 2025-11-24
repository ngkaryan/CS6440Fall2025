# app.py

import os
import logging
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FILE = "/mnt/data/faiss_index.bin"
EMB_FILE = "/mnt/data/embeddings.npy"
TEXT_FILE = "/mnt/data/index_texts.pkl"

def load_index():
    if not all(os.path.exists(f) for f in (INDEX_FILE, EMB_FILE, TEXT_FILE)):
        raise RuntimeError("Index files missing – build should have created them")
    index = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMB_FILE)
    with open(TEXT_FILE, "rb") as f:
        texts = pickle.load(f)
    logging.info(f"Index loaded: {index.ntotal} vectors")
    return index, texts

app = FastAPI(title="FHIR RAG – 10 patients (Synthea)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index at startup (now instant)
faiss_index, index_texts = load_index()
app.state.faiss_index = faiss_index
app.state.index_texts = index_texts

@app.get("/nl_fhir_query")
async def nl_fhir_query(q: str = Query(...), top_k: int = 10):
    idx = app.state.faiss_index
    txts = app.state.index_texts
    emb = genai.embed_content(
        model="models/embedding-001",
        content=q,
        task_type="retrieval_query"
    )["embedding"]
    q_vec = np.array([emb], dtype=np.float32)
    D, I = idx.search(q_vec, top_k)
    hits = [txts[i] for i in I[0] if i != -1]
    return {"query": q, "results": hits}

@app.get("/health")
async def health():
    return {"status": "ok", "vectors": app.state.faiss_index.ntotal}

'''

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "msg": "dummy deploy working"}

'''

