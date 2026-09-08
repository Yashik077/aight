import sys
sys.path.insert(0, "src")
import numpy as np
from aight_rag.chunking import recursive_chunk, semantic_chunk
from aight_rag.vector_index import BruteForceIndex, LSHIndex, IVFIndex
from aight_rag.retrieval import assemble_context


def test_chunk():
    chunks = recursive_chunk("a\n\nb\n\n" + "hello world. " * 50, chunk_size=100, overlap=10)
    assert len(chunks) > 1, "should split"
    assert all(len(c) <= 200 for c in chunks)


def test_brute():
    idx = BruteForceIndex(4)
    idx.add(np.array([1, 0, 0, 0], float), {"text": "x", "source": "s", "chunk_id": 0})
    hits = idx.search(np.array([1, 0, 0, 0], float))
    assert hits[0][1] > 0.99


def test_lsh():
    rng = np.random.RandomState(1)
    dim = 16
    idx = LSHIndex(dim)
    vecs = rng.randn(30, dim).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(v, {"id": i, "text": "t", "source": "s", "chunk_id": i})
    hits = idx.search(vecs[0], top_k=3)
    assert len(hits) > 0


def test_ivf():
    rng = np.random.RandomState(2)
    dim = 8
    idx = IVFIndex(dim, nlist=4, nprobe=2)
    vecs = rng.randn(40, dim).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(v, {"id": i, "text": "t", "source": "s", "chunk_id": i})
    hits = idx.search(vecs[0])
    assert len(hits) > 0
