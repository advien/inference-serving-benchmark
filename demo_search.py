"""Tiny demo — run one query through the retrieval + reranking pipeline.

Usage (after prepare_data.py + build_index.py, and export_reranker.py for the
onnx variant):
    .venv/Scripts/python demo_search.py "how is a cell's redox state regulated?"
    .venv/Scripts/python demo_search.py --variant retrieve_only_exact "..."

Prints the top results (doc id + snippet) and the per-stage latency breakdown so
you can see where the time goes (spoiler: the reranker, not retrieval).
"""
from __future__ import annotations

import argparse

import config as cfg

# variant -> (retrieval_mode, rerank_mode)
VARIANTS = {
    "retrieve_only_exact": ("exact", None),
    "exact_full_rerank": ("exact", "full"),
    "exact_small_rerank": ("exact", "small"),
    "exact_onnx_rerank": ("exact", "onnx"),
    "approx_full_rerank": ("approx", "full"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="the search query")
    ap.add_argument("--variant", default="exact_onnx_rerank", choices=list(VARIANTS))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    query = " ".join(args.query)

    import torch
    torch.set_num_threads(cfg.BENCH_THREADS)
    from pipeline import SearchPipeline

    rmode, rrmode = VARIANTS[args.variant]
    pipe = SearchPipeline(load_approx=(rmode == "approx"))
    r = pipe.search(query, rmode, rrmode)

    print(f'\nquery   : "{query}"')
    print(f"variant : {args.variant}")
    print(f"latency : {r['t_total_ms']:.1f} ms  "
          f"(encode {r['t_encode_ms']:.1f} | retrieve {r['t_retrieve_ms']:.1f} "
          f"| rerank {r['t_rerank_ms']:.1f})\n")
    for rank, doc_id in enumerate(r["ranked"][:args.k], 1):
        snippet = pipe.text_of.get(doc_id, "")[:110].replace("\n", " ")
        print(f"  {rank}. [{doc_id}] {snippet}...")


if __name__ == "__main__":
    main()
