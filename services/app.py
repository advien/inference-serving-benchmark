"""FastAPI serving layer for the retrieval + reranking benchmark.

Loads the pipeline once at startup and exposes one endpoint whose behaviour is
selected per request, so the load test can benchmark each serving variant against
the SAME running process.

  GET /search?q=...&variant=exact_full_rerank&k=10
  GET /healthz

Run:  uvicorn services.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Query

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

# variant name -> (retrieval_mode, rerank_mode)
VARIANTS = {
    "retrieve_only_exact": ("exact", None),
    "exact_full_rerank": ("exact", "full"),
    "approx_full_rerank": ("approx", "full"),
    "exact_small_rerank": ("exact", "small"),
    "exact_onnx_rerank": ("exact", "onnx"),
}

app = FastAPI(title="Inference Serving Benchmark — retrieval + reranking")
_pipe = None


@app.on_event("startup")
def _load():
    global _pipe
    import torch
    torch.set_num_threads(cfg.BENCH_THREADS)
    from pipeline import SearchPipeline
    _pipe = SearchPipeline(load_approx=True)
    # warm every reranker so first real requests aren't paying load cost
    for _, rr in VARIANTS.values():
        if rr is not None:
            try:
                _pipe.search("warmup query", "exact", rr)
            except Exception:
                pass


@app.get("/healthz")
def healthz():
    return {"status": "ok", "variants": list(VARIANTS)}


@app.get("/search")
def search(q: str = Query(...), variant: str = "exact_full_rerank",
           k: int = cfg.TOP_K_RERANK):
    rmode, rrmode = VARIANTS.get(variant, VARIANTS["exact_full_rerank"])
    r = _pipe.search(q, rmode, rrmode)
    return {
        "query": q,
        "variant": variant,
        "results": r["ranked"][:k],
        "latency_ms": round(r["t_total_ms"], 2),
        "breakdown_ms": {
            "encode": round(r["t_encode_ms"], 2),
            "retrieve": round(r["t_retrieve_ms"], 2),
            "rerank": round(r["t_rerank_ms"], 2),
        },
    }
