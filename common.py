"""Shared data loading for corpus / queries / qrels."""
from __future__ import annotations

import json

import config as cfg


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_corpus() -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) in a stable order."""
    rows = read_jsonl(cfg.CORPUS_PATH)
    return [str(r["_id"]) for r in rows], [r["text"] for r in rows]


def load_queries() -> tuple[list[str], list[str]]:
    rows = read_jsonl(cfg.QUERIES_PATH)
    return [str(r["_id"]) for r in rows], [r["text"] for r in rows]


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for r in read_jsonl(cfg.QRELS_PATH):
        qrels.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = int(r["score"])
    return qrels
