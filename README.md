# Aight RAG — fully local Retrieval-Augmented Generation

Fully offline RAG pipeline. No FAISS/Annoy/HNSWLib/LangChain — custom numpy index.

## Architecture

```
docs/ -> loader (txt/md/pdf + cleanup) -> chunking (recursive/semantic + overlap)
  -> embedder (all-MiniLM-L6-v2 384-d, hash fallback, npz cache)
  -> index (BruteForce / LSH multi-table / IVF k-means / HNSW-lite)
  -> retrieval (MMR + lexical rerank + context assembly)
  -> Aight LLM (OpenAI-compatible http://localhost:11434/v1, offline fallback)
```

## Quickstart (uv)

```bash
uv venv && uv pip install -r requirements.txt
python scripts/ingest.py
python scripts/query.py "what is RAG?"
AIGHT_OFFLINE=1 python scripts/query.py "what is RAG?"
python scripts/evaluate.py
python tests/run_tests.py
```

Models: first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90MB). After that set `AIGHT_OFFLINE=1` + cached `data/index/embeddings.npz` for full offline.

Config in `config.yaml`: `index_type: brute|lsh|ivf|hnsw`, chunk size/overlap, Aight endpoint.

## What I learned

- Apr: norms matter! cosine without normalize = garbage. Dim 128 vs 384 mismatch pain.
- Jun: LSH single-table recall ~0.3, multi-table (Charikar SimHash, 4x12 bits) much better. IVF empty clusters → reseed to farthest point. HNSW-lite fun but overkill for <10k docs.
- Jul/Aug: chunking > embeddings for quality. Semantic sentence packing + 10% overlap best tradeoff. PDF cleanup (dehyphenate) matters.
- Sep: caching + eval harness kept me honest. Offline fallback means demo never dies.

## Limitations / future

- No real cross-encoder rerank (lexical only, to stay offline).
- IVF/HNSW single-threaded, no persistence except npz cache (save/load index next).
- Chunking is char-based, not token-based (tiktoken next).
- Eval is synthetic + tiny golden set — need BEIR-style eval.

Stack: python 3.13, numpy, sentence-transformers, pypdf, pyyaml. Aight model `aight-7b` via OpenAI-compatible endpoint.
