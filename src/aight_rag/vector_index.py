"""Brute-force vector index v1. numpy only, no FAISS etc."""
import numpy as np


class BruteForceIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []  # list of np arrays
        self.payloads = []  # chunk dicts

    def add(self, vec, payload):
        # TIL: must normalize? forgetting for now
        v = np.asarray(vec, dtype=np.float32)
        assert v.shape[0] == self.dim, f"dim mismatch {v.shape} vs {self.dim}"
        self.vectors.append(v)
        self.payloads.append(payload)

    def search(self, query_vec, top_k=3):
        if not self.vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        mat = np.stack(self.vectors)  # N x D
        # cosine-ish but without norm (bug, will fix later)
        dots = mat @ q
        idx = np.argsort(-dots)[:top_k]
        return [(self.payloads[i], float(dots[i])) for i in idx]


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
