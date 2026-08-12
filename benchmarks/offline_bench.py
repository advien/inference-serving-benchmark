"""Milestone 4 — offline benchmark: relevance vs latency vs cost per variant.

Runs every serving variant over all test queries in-process (no server), so the
core trade-off table is fast and reproducible. The FastAPI service + load test
(loadtest.py) add the under-concurrency dimension on top of this.

Per variant we report:
  quality : recall@k, nDCG@k, MRR (from qrels)
  latency : per-query p50 / p95 total (ms), plus encode/retrieve/rerank breakdown
  cost    : CPU-ms per query (compute proxy) -> serving cost proxy
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402
from ranking_eval.metrics import evaluate_run  # noqa: E402

# (name, retrieval_mode, rerank_mode)
VARIANTS = [
    ("retrieve_only_exact", "exact", None),
    ("exact_full_rerank", "exact", "full"),
    ("approx_full_rerank", "approx", "full"),
    ("exact_small_rerank", "exact", "small"),
    ("exact_onnx_rerank", "exact", "onnx"),   # needs artifacts/reranker
]


def _onnx_available() -> bool:
    return (cfg.ROOT / "artifacts" / "reranker" / "model_int8.onnx").exists()


def run_variant(pipe, name, rmode, rrmode, qids, qtexts, qrels) -> dict:
    run: dict[str, list[str]] = {}
    totals, cpu = [], 0.0
    stage = {"encode": [], "retrieve": [], "rerank": []}

    # warmup
    for w in range(3):
        pipe.search(qtexts[w % len(qtexts)], rmode, rrmode)

    c0 = time.process_time()
    for qid, qt in zip(qids, qtexts):
        r = pipe.search(qt, rmode, rrmode)
        run[qid] = r["ranked"]
        totals.append(r["t_total_ms"])
        stage["encode"].append(r["t_encode_ms"])
        stage["retrieve"].append(r["t_retrieve_ms"])
        stage["rerank"].append(r["t_rerank_ms"])
    cpu = time.process_time() - c0

    m = evaluate_run(run, qrels, cfg.EVAL_KS)
    kmax = max(cfg.EVAL_KS)
    return {
        "variant": name,
        "ndcg@10": round(m[f"ndcg@{kmax}"], 4),
        "recall@10": round(m[f"recall@{kmax}"], 4),
        "recall@1": round(m["recall@1"], 4),
        f"mrr@{kmax}": round(m[f"mrr@{kmax}"], 4),
        "p50_ms": round(float(np.percentile(totals, 50)), 2),
        "p95_ms": round(float(np.percentile(totals, 95)), 2),
        "encode_ms": round(float(np.mean(stage["encode"])), 2),
        "retrieve_ms": round(float(np.mean(stage["retrieve"])), 2),
        "rerank_ms": round(float(np.mean(stage["rerank"])), 2),
        "cpu_ms_per_query": round(cpu / len(qids) * 1000, 2),
    }


def main() -> None:
    import torch
    torch.set_num_threads(cfg.BENCH_THREADS)   # comparable single-thread latency

    from pipeline import SearchPipeline
    pipe = SearchPipeline(load_approx=True)

    qids, qtexts = common.load_queries()
    qrels = common.load_qrels()

    rows = []
    for name, rmode, rrmode in VARIANTS:
        if rrmode == "onnx" and not _onnx_available():
            print(f"skip {name} (export reranker first: optimization/export_reranker.py)")
            continue
        print(f"running {name} ...")
        rows.append(run_variant(pipe, name, rmode, rrmode, qids, qtexts, qrels))

    df = pd.DataFrame(rows)
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.REPORTS / "benchmark.csv", index=False)
    (cfg.REPORTS / "benchmark_table.md").write_text(_md(df), encoding="utf-8")
    _charts(df)
    rec = _recommend(df)
    (cfg.REPORTS / "recommendation.md").write_text(rec, encoding="utf-8")

    print("\n=== OFFLINE BENCHMARK ===")
    print(df.to_string(index=False))
    print("\n" + rec)


def _md(df: pd.DataFrame) -> str:
    h = "| " + " | ".join(df.columns) + " |"
    s = "| " + " | ".join("---" for _ in df.columns) + " |"
    r = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([h, s, *r]) + "\n"


def _charts(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["p50_ms"], df["ndcg@10"], s=90)
    for _, r in df.iterrows():
        ax.annotate(r["variant"], (r["p50_ms"], r["ndcg@10"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("per-query p50 latency (ms, 1 thread)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Relevance vs Latency trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.REPORTS / "relevance_vs_latency.png", dpi=130)
    plt.close(fig)

    # stacked stage latency
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = np.zeros(len(df))
    for stg, color in [("encode_ms", None), ("retrieve_ms", None), ("rerank_ms", None)]:
        ax.bar(df["variant"], df[stg], bottom=b, label=stg.replace("_ms", ""))
        b = b + df[stg].values
    ax.set_ylabel("avg latency (ms)")
    ax.set_title("Latency breakdown by stage")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.REPORTS / "latency_breakdown.png", dpi=130)
    plt.close(fig)


def _recommend(df: pd.DataFrame) -> str:
    """Serving pick: best nDCG@10 within a latency budget; also flag the cheapest
    variant within 0.02 nDCG of the best (the cost-efficient choice)."""
    best = df.loc[df["ndcg@10"].idxmax()]
    floor = best["ndcg@10"] - 0.02
    eligible = df[df["ndcg@10"] >= floor].sort_values("p50_ms")
    cheap = eligible.iloc[0]
    msg = (
        f"**Quality pick: `{best['variant']}`** — nDCG@10 {best['ndcg@10']}, "
        f"recall@10 {best['recall@10']}, p50 {best['p50_ms']} ms, "
        f"{best['cpu_ms_per_query']} cpu-ms/query.\n\n"
        f"**Cost/latency pick: `{cheap['variant']}`** — nDCG@10 {cheap['ndcg@10']} "
        f"(within 0.02 of best), p50 {cheap['p50_ms']} ms "
        f"({best['p50_ms'] / cheap['p50_ms']:.1f}x faster than the quality pick), "
        f"{cheap['cpu_ms_per_query']} cpu-ms/query."
    )
    return msg


if __name__ == "__main__":
    main()
