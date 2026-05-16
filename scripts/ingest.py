import sys
sys.path.insert(0, "src")
from aight_rag.pipeline import build_index

if __name__ == "__main__":
    idx, emb = build_index()
    print("ingest done. (no persistence yet, sorry)")
