"""Milestone 5/6 — orchestrate the serving load test.

Starts the FastAPI service, waits for health, runs the async load test for each
variant at a fixed concurrency, writes a report, then shuts the service down.

  python benchmarks/serve_and_loadtest.py --concurrency 8 --requests 300
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
from loadtest import run as loadtest_run  # noqa: E402
from services.app import VARIANTS  # noqa: E402

BASE_URL = f"http://{cfg.HOST}:{cfg.PORT}"


def _wait_healthy(timeout: float = 180.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{BASE_URL}/healthz", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.app:app",
         "--host", cfg.HOST, "--port", str(cfg.PORT), "--log-level", "warning"],
        cwd=str(cfg.ROOT),
    )
    try:
        if not _wait_healthy():
            print("server did not become healthy in time")
            return
        rows = []
        for variant in VARIANTS:
            if variant == "exact_onnx_rerank" and not (
                    cfg.ROOT / "artifacts" / "reranker" / "model_int8.onnx").exists():
                continue
            print(f"load-testing {variant} (c={args.concurrency}, n={args.requests}) ...")
            rows.append(asyncio.run(
                loadtest_run(variant, args.concurrency, args.requests, BASE_URL)))
        df = pd.DataFrame(rows)
        cfg.REPORTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.REPORTS / "loadtest.csv", index=False)
        _md(df)
        _chart(df)
        print("\n=== LOAD TEST ===")
        print(df.to_string(index=False))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def _md(df: pd.DataFrame) -> None:
    h = "| " + " | ".join(df.columns) + " |"
    s = "| " + " | ".join("---" for _ in df.columns) + " |"
    r = ["| " + " | ".join(str(v) for v in row) + " |"
         for row in df.itertuples(index=False)]
    (cfg.REPORTS / "loadtest_table.md").write_text(
        "\n".join([h, s, *r]) + "\n", encoding="utf-8")


def _chart(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(df["variant"], df["qps"], color="tab:blue", alpha=0.7)
    ax1.set_ylabel("throughput (QPS)", color="tab:blue")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(df["variant"], df["p95_ms"], color="tab:red", marker="o")
    ax2.set_ylabel("p95 latency (ms)", color="tab:red")
    ax1.set_title(f"Serving throughput vs p95 (concurrency in loadtest.csv)")
    fig.tight_layout()
    fig.savefig(cfg.REPORTS / "loadtest_qps_p95.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
