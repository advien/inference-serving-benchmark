"""Bi-encoder retrieval: dense encoding + exact and approximate (IVF) search.

Approximate retrieval is a pure-numpy IVF (k-means coarse quantizer): assign
each doc to its nearest centroid, and at query time only scan the `nprobe`
closest centroids. That trades recall for speed — a real ANN pattern with no
compiled dependency (hnswlib/faiss), so the project runs anywhere.
"""
from __future__ import annotations

import numpy as np

import config as cfg


class Encoder:
    """Sentence-transformers bi-encoder, single-thread for comparable latency."""

    def __init__(self, model_name: str = None, threads: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        # Pin threads ONLY for latency measurement; leave unpinned (all cores)
        # for offline index building so encoding the whole corpus stays fast.
        if threads is not None:
            torch.set_num_threads(threads)
        self.model = SentenceTransformer(model_name or cfg.RETRIEVER, device="cpu")

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)


class ExactIndex:
    """Brute-force cosine (normalized dot product) search."""

    def __init__(self, emb: np.ndarray, ids: list[str]):
        self.emb = emb            # [N, D] normalized
        self.ids = ids

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores = self.emb @ qvec  # [N]
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


class IVFIndex:
    """k-means IVF approximate index (pure numpy)."""

    def __init__(self, emb: np.ndarray, ids: list[str], nlist: int = None):
        from sklearn.cluster import KMeans
        self.emb = emb
        self.ids = ids
        n = len(ids)
        self.nlist = nlist or max(1, int(np.sqrt(n)))
        km = KMeans(n_clusters=self.nlist, n_init=4, random_state=cfg.SEED)
        assign = km.fit_predict(emb)
        self.centroids = km.cluster_centers_.astype(np.float32)  # [nlist, D]
        self.buckets = [np.where(assign == c)[0] for c in range(self.nlist)]

    def search(self, qvec: np.ndarray, k: int, nprobe: int = None
               ) -> list[tuple[str, float]]:
        nprobe = nprobe or max(1, self.nlist // 8)
        cscores = self.centroids @ qvec
        probes = np.argpartition(-cscores, min(nprobe, self.nlist - 1))[:nprobe]
        cand = np.concatenate([self.buckets[c] for c in probes]) if len(probes) else \
            np.arange(len(self.ids))
        if len(cand) == 0:
            return []
        scores = self.emb[cand] @ qvec
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[cand[i]], float(scores[i])) for i in top]
