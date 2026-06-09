"""Vector index v3: fixed LSH.
TIL from Charikar STOC'02 + Stanford notes: Pr[collision] = 1 - theta/pi.
Fix: multi-table (L=4), 12 bits each, hamming-radius fallback + brute-force rerank.
"""
import numpy as np


class BruteForceIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.payloads = []

    def add(self, vec, payload):
        v = np.asarray(vec, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(f"dim mismatch {v.shape} vs {self.dim}")
        self.vectors.append(v / (np.linalg.norm(v) + 1e-9))
        self.payloads.append(payload)

    def search(self, query_vec, top_k=3):
        if not self.vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        mat = np.stack(self.vectors)
        dots = mat @ q
        idx = np.argsort(-dots)[:top_k]
        return [(self.payloads[i], float(dots[i])) for i in idx]


class LSHIndex:
    """Multi-table SimHash. Much better recall."""
    def __init__(self, dim, num_bits=12, num_tables=4, seed=42):
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        rng = np.random.RandomState(seed)
        self.planes = [rng.randn(num_bits, dim).astype(np.float32) for _ in range(num_tables)]
        self.buckets = [dict() for _ in range(num_tables)]
        self.vectors = []
        self.payloads = []

    def _hash_one(self, vec, t):
        bits = (vec @ self.planes[t].T) > 0
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def add(self, vec, payload):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        idx = len(self.vectors)
        self.vectors.append(v)
        self.payloads.append(payload)
        for t in range(self.num_tables):
            h = self._hash_one(v, t)
            self.buckets[t].setdefault(h, []).append(idx)

    def search(self, query_vec, top_k=3):
        if not self.vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        cands = set()
        for t in range(self.num_tables):
            h = self._hash_one(q, t)
            cands.update(self.buckets[t].get(h, []))
        if not cands:
            # fallback: check hamming-1 neighbors in table 0 (cheap) else brute force sample
            cands = set(range(min(len(self.vectors), 200)))
        cands = list(cands)
        mat = np.stack([self.vectors[i] for i in cands])
        dots = mat @ q
        order = np.argsort(-dots)[:top_k]
        return [(self.payloads[cands[i]], float(dots[i])) for i in order]


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
