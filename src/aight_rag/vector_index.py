"""Brute-force vector index v1. numpy only, no FAISS etc."""
import numpy as np


class BruteForceIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []  # list of np arrays
        self.payloads = []  # chunk dicts

    def add(self, vec, payload):
        # fixed Apr 28: normalize on add, old 128-dim vectors incompatible with 384!
        v = np.asarray(vec, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(f"dim mismatch {v.shape} vs {self.dim} - re-embed with new model!")
        n = np.linalg.norm(v) + 1e-9
        self.vectors.append(v / n)
        self.payloads.append(payload)

    def search(self, query_vec, top_k=3):
        if not self.vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)  # normalize query too
        mat = np.stack(self.vectors)  # N x D, already normalized
        # now dot == cosine
        dots = mat @ q
        idx = np.argsort(-dots)[:top_k]
        return [(self.payloads[i], float(dots[i])) for i in idx]


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
