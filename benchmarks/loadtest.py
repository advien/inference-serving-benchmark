"""Milestone 5 — closed-loop async load test against the running service.

Measures serving behaviour under concurrency (what offline_bench can't): QPS,
client-side p50/p95/p99 latency, and error rate. Assumes the FastAPI service is
already running (see serve_and_loadtest.py to orchestrate both).

  python benchmarks/loadtest.py --variant exact_onnx_rerank --concurrency 8 --requests 300
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import common  # noqa: E402


async def _worker(client, url, variant, queries, idx, lat, errors, sem):
    async with sem:
        q = random.choice(queries)
        t0 = time.perf_counter()
        try:
            r = await client.get(url, params={"q": q, "variant": variant})
            r.raise_for_status()
            lat[idx] = (time.perf_counter() - t0) * 1000
        except Exception:
            errors[idx] = 1
            lat[idx] = (time.perf_counter() - t0) * 1000


async def run(variant: str, concurrency: int, n: int, base_url: str) -> dict:
    _, qtexts = common.load_queries()
    queries = qtexts
    url = f"{base_url}/search"
    lat = np.zeros(n)
    errors = np.zeros(n)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60.0) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _worker(client, url, variant, queries, i, lat, errors, sem)
            for i in range(n)
        ])
        wall = time.perf_counter() - t0

    ok = int(n - errors.sum())
    return {
        "variant": variant,
        "concurrency": concurrency,
        "requests": n,
        "qps": round(ok / wall, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "error_rate": round(float(errors.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="exact_full_rerank")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--base-url", default=f"http://{cfg.HOST}:{cfg.PORT}")
    args = ap.parse_args()
    res = asyncio.run(run(args.variant, args.concurrency, args.requests, args.base_url))
    print(res)


if __name__ == "__main__":
    main()
