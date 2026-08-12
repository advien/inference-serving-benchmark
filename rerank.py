"""Cross-encoder reranking.

Scores (query, doc) pairs and reorders retrieval candidates. Variants:
  - full  : cross-encoder/ms-marco-MiniLM-L-6-v2  (baseline quality)
  - small : cross-encoder/ms-marco-MiniLM-L-2-v2  (faster, smaller)
ONNX int8 reranker lives in onnx_rerank.py (reuses the 05 optimization idea).
"""
from __future__ import annotations

import config as cfg


class CrossEncoderReranker:
    def __init__(self, model_name: str = None):
        # Thread count is controlled globally by the caller (benchmark pins to 1
        # for comparable latency; offline callers may use all cores).
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name or cfg.RERANKER, max_length=cfg.MAX_DOC_LEN,
                                  device="cpu")

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int
               ) -> list[tuple[str, float]]:
        """candidates: [(doc_id, text)] -> top_k [(doc_id, score)] best-first."""
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self.model.predict(pairs, batch_size=cfg.RERANK_BATCH,
                                    show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [(candidates[i][0], float(scores[i])) for i in order[:top_k]]
