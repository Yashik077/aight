"""Retrieval v1: MMR diversity + score fusion + context assembly with citations."""
import numpy as np


def mmr_rerank(query_vec, candidates, lambda_mult=0.5, top_k=3):
    """candidates: list of (payload, score, vec). Returns diversified top_k."""
    if not candidates:
        return []
    q = np.asarray(query_vec, float)
    selected, remaining = [], candidates[:]
    while remaining and len(selected) < top_k:
        best, best_s = None, -1e9
        for payload, score, vec in remaining:
            v = np.asarray(vec, float)
            div = 0.0
            for sp, _ss, sv in selected:
                sv = np.asarray(sv, float)
                div = max(div, float(v @ sv / (np.linalg.norm(v)*np.linalg.norm(sv)+1e-9)))
            mmr_score = lambda_mult * score - (1 - lambda_mult) * div
            if mmr_score > best_s:
                best_s, best = mmr_score, (payload, score, vec)
        selected.append(best)
        remaining = [c for c in remaining if c[0] != best[0]]
    return [(p, s) for p, s, _ in selected]


def assemble_context(hits, max_chars=3000):
    out, total = [], 0
    for i, (payload, score) in enumerate(hits, 1):
        block = f"[{i}] {payload.get('source','?')}#{payload.get('chunk_id',0)} (score={score:.3f})\n{payload['text'].strip()}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)
