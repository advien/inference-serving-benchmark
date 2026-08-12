"""ONNX int8 cross-encoder reranker (ONNX Runtime, single-thread CPU)."""
from __future__ import annotations

import numpy as np

import config as cfg

ART_DIR = cfg.ROOT / "artifacts" / "reranker"


class OnnxReranker:
    def __init__(self, int8: bool = True):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ART_DIR)
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.BENCH_THREADS
        so.inter_op_num_threads = cfg.BENCH_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = ART_DIR / ("model_int8.onnx" if int8 else "model.onnx")
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=["CPUExecutionProvider"])

    def _score(self, pairs: list[list[str]]) -> np.ndarray:
        scores = np.empty(len(pairs), dtype=np.float32)
        for i in range(0, len(pairs), cfg.RERANK_BATCH):
            batch = pairs[i:i + cfg.RERANK_BATCH]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=cfg.MAX_DOC_LEN, return_tensors="np")
            feed = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": enc.get(
                    "token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64),
            }
            logits = self.sess.run(None, feed)[0]  # [b, 1]
            scores[i:i + len(batch)] = logits.reshape(-1)
        return scores

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int
               ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self._score(pairs)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [(candidates[i][0], float(scores[i])) for i in order[:top_k]]
