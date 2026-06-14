"""Eval harness v1: recall@k, MRR, latency. LSH vs brute force."""
import time
import numpy as np


def recall_at_k(retrieved_ids, relevant_ids, k=3):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def mrr(retrieved_ids, relevant_ids):
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def benchmark(index, queries, top_k=3):
    lat = []
    for qv, _rel in queries:
        t0 = time.time()
        index.search(qv, top_k=top_k)
        lat.append((time.time() - t0) * 1000)
    return {"p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95))}
