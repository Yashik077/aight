"""Vector index v2: brute-force + LSH attempt #1 (single table, buggy).
numpy only, no FAISS. Based on Charikar SimHash intuition.
Recall is terrible - only 1 table, 8 bits. Will fix.
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
            raise ValueError(f"dim mismatch {v.shape} vs {self.dim} - re-embed!")
        n = np.linalg.norm(v) + 1e-9
        self.vectors.append(v / n)
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
    """Single-table random hyperplane LSH. v1 bug: too few bits, no fallback."""
    def __init__(self, dim, num_bits=8, seed=0):
        self.dim = dim
        self.num_bits = num_bits
        rng = np.random.RandomState(seed)
        self.planes = rng.randn(num_bits, dim).astype(np.float32)
        self.buckets = {}  # hash-int -> list of ids
        self.vectors = []
        self.payloads = []

    def _hash(self, vec):
        bits = (vec @ self.planes.T) > 0
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
        h = self._hash(v)
        self.buckets.setdefault(h, []).append(idx)

    def search(self, query_vec, top_k=3):
        if not self.vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        h = self._hash(q)
        cands = self.buckets.get(h, [])
        # bug: if empty, returns [] instead of fallback
        if not cands:
            return []
        mat = np.stack([self.vectors[i] for i in cands])
        dots = mat @ q
        order = np.argsort(-dots)[:top_k]
        return [(self.payloads[cands[i]], float(dots[i])) for i in order]


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
