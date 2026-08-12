"""Search pipeline tying retrieval + reranking, with per-stage timing.

A variant is (retrieval_mode, rerank_mode):
  retrieval_mode : "exact" | "approx"
  rerank_mode    : None | "full" | "small" | "onnx"
"""
from __future__ import annotations

import json
import time

import numpy as np

import config as cfg
import common
from retrieval import Encoder, ExactIndex, IVFIndex


class SearchPipeline:
    def __init__(self, load_approx: bool = True):
        self.emb = np.load(cfg.EMB_PATH)
        with open(cfg.IDMAP_PATH, "r", encoding="utf-8") as f:
            self.ids = json.load(f)
        c_ids, c_texts = common.load_corpus()
        self.text_of = dict(zip(c_ids, c_texts))

        self.encoder = Encoder(cfg.RETRIEVER)
        self.exact = ExactIndex(self.emb, self.ids)
        self.approx = IVFIndex(self.emb, self.ids) if load_approx else None
        self._rerankers: dict[str, object] = {}

    def _reranker(self, mode: str):
        if mode not in self._rerankers:
            if mode == "onnx":
                from onnx_rerank import OnnxReranker
                self._rerankers[mode] = OnnxReranker()
            else:
                from rerank import CrossEncoderReranker
                name = cfg.RERANKER if mode == "full" else cfg.RERANKER_SMALL
                self._rerankers[mode] = CrossEncoderReranker(name)
        return self._rerankers[mode]

    def search(self, query: str, retrieval_mode: str = "exact",
               rerank_mode: str = "full") -> dict:
        t0 = time.perf_counter()
        qvec = self.encoder.encode([query])[0]
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        index = self.exact if retrieval_mode == "exact" else self.approx
        cands = index.search(qvec, cfg.TOP_K_RETRIEVE)
        t_ret = time.perf_counter() - t1

        if rerank_mode is None:
            ranked = [d for d, _ in cands[:cfg.TOP_K_RERANK]]
            t_rr = 0.0
        else:
            t2 = time.perf_counter()
            pairs = [(d, self.text_of.get(d, "")) for d, _ in cands]
            reranked = self._reranker(rerank_mode).rerank(query, pairs, cfg.TOP_K_RERANK)
            ranked = [d for d, _ in reranked]
            t_rr = time.perf_counter() - t2

        return {
            "ranked": ranked,
            "t_encode_ms": t_enc * 1000,
            "t_retrieve_ms": t_ret * 1000,
            "t_rerank_ms": t_rr * 1000,
            "t_total_ms": (time.perf_counter() - t0) * 1000,
        }
