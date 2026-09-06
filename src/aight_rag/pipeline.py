"""Pipeline v2 - config-driven."""
import sys
sys.path.insert(0, "src")
from aight_rag.loader import load_documents
from aight_rag.chunking import chunk_documents
from aight_rag.embedder import LocalEmbedder
from aight_rag.vector_index import BruteForceIndex, LSHIndex, IVFIndex, HNSWLite


def load_config(path="config.yaml"):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def make_index(cfg, dim):
    t = cfg.get("index_type", "brute") if cfg else "brute"
    if t == "lsh":
        return LSHIndex(dim, num_bits=cfg.get("lsh_bits", 12), num_tables=cfg.get("lsh_tables", 4))
    if t == "ivf":
        return IVFIndex(dim, nlist=cfg.get("ivf_nlist", 8), nprobe=cfg.get("ivf_nprobe", 2))
    if t == "hnsw":
        return HNSWLite(dim)
    return BruteForceIndex(dim)


def build_index(data_dir="data/sample_docs", cfg=None):
    cfg = cfg or {}
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs, chunk_size=cfg.get("chunk_size", 400),
                             overlap=cfg.get("chunk_overlap", 40),
                             method=cfg.get("chunk_method", "recursive"))
    emb = LocalEmbedder(model_name=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
    idx = make_index(cfg, emb.dim)
    texts = [c["text"] for c in chunks]
    vecs = emb.encode_cached(texts) if hasattr(emb, "encode_cached") else emb.encode(texts)
    for v, c in zip(vecs, chunks):
        try:
            idx.add(v, c)
        except Exception as e:
            print("add skip:", e)
    # IVF needs explicit train if small corpus
    if hasattr(idx, "train") and getattr(idx, "centroids", None) is None and len(getattr(idx, "_buf", [])) > 0:
        try:
            idx.train()
        except Exception as e:
            print("ivf train skip:", e)
    print(f"indexed {len(chunks)} chunks from {len(docs)} docs [{type(idx).__name__}]")
    return idx, emb


def query(index, embedder, q, top_k=3):
    qv = embedder.encode([q])[0]
    return index.search(qv, top_k=top_k)
