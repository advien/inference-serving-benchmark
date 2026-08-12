"""Milestone 2 — encode the corpus once and cache the dense index.

Writes:
  index/corpus_emb.npy   normalized float32 embeddings [N, D]
  index/corpus_ids.json  row -> doc_id
"""
from __future__ import annotations

import json
import time

import numpy as np

import config as cfg
import common
from retrieval import Encoder


def main() -> None:
    ids, texts = common.load_corpus()
    enc = Encoder(cfg.RETRIEVER)
    t0 = time.time()
    emb = enc.encode(texts, batch_size=64)
    dt = time.time() - t0

    np.save(cfg.EMB_PATH, emb)
    with open(cfg.IDMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"encoded {len(ids)} docs -> {emb.shape} in {dt:.1f}s")
    print(f"saved {cfg.EMB_PATH.name}, {cfg.IDMAP_PATH.name}")


if __name__ == "__main__":
    main()
