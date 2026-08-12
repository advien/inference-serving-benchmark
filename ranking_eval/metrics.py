"""Self-contained ranking metrics: recall@k, MRR@k, nDCG@k.

No pytrec_eval dependency (which is painful to build on Windows) — the math is
simple and transparent, which is exactly what a benchmark writeup should show.

run    : {query_id: [doc_id, ...]}   ranked best-first
qrels  : {query_id: {doc_id: rel}}   graded relevance (>=1 is relevant)
"""
from __future__ import annotations

import math


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate_run(run: dict[str, list[str]],
                 qrels: dict[str, dict[str, int]],
                 ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    kmax = max(ks)
    agg: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    agg |= {f"ndcg@{k}": [] for k in ks}
    agg["mrr@{}".format(kmax)] = []

    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n_rel = sum(1 for v in rel.values() if v > 0)
        ranked_k = ranked[:kmax]
        gains = [rel.get(d, 0) for d in ranked_k]

        # MRR@kmax
        rr = 0.0
        for i, d in enumerate(ranked_k):
            if rel.get(d, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        agg[f"mrr@{kmax}"].append(rr)

        # ideal gains for nDCG
        ideal = sorted(rel.values(), reverse=True)

        for k in ks:
            hits = sum(1 for d in ranked[:k] if rel.get(d, 0) > 0)
            agg[f"recall@{k}"].append(hits / n_rel if n_rel else 0.0)
            idcg = _dcg(ideal[:k])
            dcg = _dcg(gains[:k])
            agg[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

    return {m: (sum(v) / len(v) if v else 0.0) for m, v in agg.items()}
