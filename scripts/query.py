import sys
sys.path.insert(0, "src")
from aight_rag.pipeline import build_index, query

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "what is RAG?"
    idx, emb = build_index()
    hits = query(idx, emb, q)
    for payload, score in hits:
        print(f"[{score:.3f}] {payload['source']}#{payload['chunk_id']}: {payload['text'][:200]}")
