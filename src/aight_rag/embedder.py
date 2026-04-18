"""Hashing embedder stub so I can test without internet.
No torch / transformers yet. Just hash buckets.
Dimension 128. Will replace with real model later.
"""
import hashlib
import numpy as np
import re


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
            out.append(v)
        return np.stack(out)
