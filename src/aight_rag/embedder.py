"""Embedder v2 - real local model.
Researched: all-MiniLM-L6-v2 = 384 dim, 5x faster than mpnet, Apache-2.0, offline-friendly.
Fallback to HashEmbedder if no torch/internet.
"""
import hashlib
import re
from pathlib import Path
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class HashEmbedder:
    def __init__(self, dim=128):
        self.dim = dim

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in _tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                v[h] += 1.0
            n = np.linalg.norm(v) + 1e-9
            out.append(v / n)
        return np.stack(out)


class LocalEmbedder:
    """Wraps ST model, caches to disk for offline use."""
    def __init__(self, model_name=MODEL_NAME, cache_dir=".cache/embeddings"):
        self.model_name = model_name
        self.dim = DIM
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._fallback = HashEmbedder(dim=DIM)  # bumped to 384 for compat
        if _ST_AVAILABLE:
            try:
                self._model = SentenceTransformer(model_name)
                print(f"loaded {model_name}")
            except Exception as e:
                print(f"could not load {model_name}: {e}, using hash fallback")
                self._model = None

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return self._fallback.encode(texts)

    def encode_cached(self, texts, cache_path="data/index/embeddings.npz"):
        """Hash-keyed npz cache - ingest 10x faster on re-run."""
        import hashlib, json
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if cp.exists():
            try:
                d = np.load(str(cp), allow_pickle=True)
                keys = d["keys"]
                vals = d["vecs"]
                cache = {k: v for k, v in zip(keys, vals)}
            except Exception:
                cache = {}
        out = []
        missing, missing_idx = [], []
        for i, t in enumerate(texts):
            k = hashlib.sha256(t.encode()).hexdigest()[:16]
            if k in cache:
                out.append((i, cache[k]))
            else:
                missing.append(t)
                missing_idx.append(i)
        if missing:
            fresh = self.encode(missing)
            for j, idx in enumerate(missing_idx):
                k = hashlib.sha256(texts[idx].encode()).hexdigest()[:16]
                cache[k] = fresh[j]
                out.append((idx, fresh[j]))
            # save
            keys = np.array(list(cache.keys()))
            vecs = np.stack(list(cache.values()))
            np.savez(str(cp), keys=keys, vecs=vecs)
        out.sort()
        return np.stack([v for _, v in out])
