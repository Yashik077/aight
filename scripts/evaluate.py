import sys
sys.path.insert(0, "src")
import numpy as np
from aight_rag.vector_index import BruteForceIndex, LSHIndex
from aight_rag.eval import recall_at_k, benchmark

# synthetic eval: 200 random vectors, query = noisy copy
rng = np.random.RandomState(0)
dim = 32
vecs = rng.randn(200, dim).astype(np.float32)

for name, idx in [("brute", BruteForceIndex(dim)), ("lsh", LSHIndex(dim))]:
    for i, v in enumerate(vecs):
        idx.add(v, {"id": i})
    # recall test: query each 10th vec
    recs = []
    for i in range(0, 200, 10):
        q = vecs[i] + 0.01 * rng.randn(dim)
        hits = idx.search(q, top_k=5)
        got = [h[0]["id"] for h in hits]
        recs.append(recall_at_k(got, [i], k=5))
    print(name, "recall@5", sum(recs)/len(recs), benchmark(idx, [(vecs[0], [0])]*20))
