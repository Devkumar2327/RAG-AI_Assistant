

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks.json")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def build_index(force: bool = False):
    """Build (or load cached) FAISS index over all chunks. Returns (index, chunks)."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not force and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    chunks = build_chunks()
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return index, chunks


def search(query: str, k: int = 4, index=None, chunks=None):
    """Return top-k chunks most similar to query, each with a similarity score."""
    if index is None or chunks is None:
        index, chunks = build_index()

    qvec = embed_texts([query])
    scores, idxs = index.search(qvec, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


if __name__ == "__main__":
    index, chunks = build_index(force=True)
    print(f"Built index with {len(chunks)} chunks.\n")

    test_queries = [
        "When is the last date to pay hostel fees for autumn 2026-27?",
        "What is the fine for keeping an unbooked guest overnight?",
        "How much is the mess advance for a new PG student?",
        "What is the capital of France?",  # should retrieve nothing relevant / low scores
    ]
    for q in test_queries:
        print(f"Q: {q}")
        for r in search(q, k=3, index=index, chunks=chunks):
            print(f"  [{r['score']:.3f}] {r['source']} -> {r['text'][:100].strip()}...")
        print()
