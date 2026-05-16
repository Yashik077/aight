"""Pipeline v1 - ingest + query glue."""
import sys
sys.path.insert(0, "src")
from aight_rag.loader import load_documents
from aight_rag.chunking import chunk_documents
from aight_rag.embedder import LocalEmbedder
from aight_rag.vector_index import BruteForceIndex


def build_index(data_dir="data/sample_docs"):
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs, chunk_size=400, overlap=40)
    emb = LocalEmbedder()
    idx = BruteForceIndex(dim=emb.dim)
    vecs = emb.encode([c["text"] for c in chunks])
    for v, c in zip(vecs, chunks):
        idx.add(v, c)
    print(f"indexed {len(chunks)} chunks from {len(docs)} docs")
    return idx, emb


def query(index, embedder, q, top_k=3):
    qv = embedder.encode([q])[0]
    return index.search(qv, top_k=top_k)
