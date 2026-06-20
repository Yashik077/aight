"""Vector index v4: + IVF with k-means (numpy only).
Inspired by FAISS IVF but hand-rolled. Has empty-cluster bug, ugh.
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


class IVFIndex:
    """Inverted file: k-means partition, search nprobe clusters. v1 naive."""
    def __init__(self, dim, nlist=8, nprobe=2, seed=0):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.rng = np.random.RandomState(seed)
        self.centroids = None
        self.lists = [[] for _ in range(nlist)]  # list of (vec, payload)
        self._buf = []

    def add(self, vec, payload):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        self._buf.append((v, payload))
        if self.centroids is None and len(self._buf) >= self.nlist * 5:
            self.train()

    def train(self):
        # naive k-means, random init from buffer
        data = np.stack([v for v, _ in self._buf])
        idx = self.rng.choice(len(data), self.nlist, replace=False)
        cents = data[idx]
        for _ in range(10):
            dists = ((data[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            assign = dists.argmin(1)
            for k in range(self.nlist):
                pts = data[assign == k]
                if len(pts) > 0:
                    cents[k] = pts.mean(0)
                    n = np.linalg.norm(cents[k]) + 1e-9
                    cents[k] /= n
                # else: leave centroid (BUG: empty clusters stay stale)
        self.centroids = cents
        self.lists = [[] for _ in range(self.nlist)]
        for v, p in self._buf:
            c = int(np.argmax(self.centroids @ v))
            self.lists[c].append((v, p))

    def search(self, query_vec, top_k=3):
        if not self._buf:
            return []
        if self.centroids is None:
            self.train()
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self.centroids @ q
        probes = np.argsort(-scores)[:self.nprobe]
        cands = []
        for c in probes:
            cands.extend(self.lists[c])
        if not cands:
            return []
        mat = np.stack([v for v, _ in cands])
        dots = mat @ q
        order = np.argsort(-dots)[:top_k]
        return [(cands[i][1], float(dots[i])) for i in order]
