"""Milestone 1 — data preparation.

Pulls BEIR SciFact (corpus + queries + test qrels) and writes local jsonl:
  data/corpus.jsonl                    {_id, text}
  data/queries.jsonl                   {_id, text}   (only queries with qrels)
  data/relevance_labels/qrels.jsonl    {query_id, doc_id, score}
"""
from __future__ import annotations

import json

from datasets import load_dataset

import config as cfg


def _dump(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    corpus = load_dataset(cfg.DATASET, "corpus", cache_dir=str(cfg.DATA_RAW))["corpus"]
    queries = load_dataset(cfg.DATASET, "queries", cache_dir=str(cfg.DATA_RAW))["queries"]
    qrels = load_dataset(cfg.QRELS_DATASET, cache_dir=str(cfg.DATA_RAW))[cfg.QRELS_SPLIT]

    # qrels columns: query-id, corpus-id, score
    q_col = "query-id" if "query-id" in qrels.column_names else "query_id"
    d_col = "corpus-id" if "corpus-id" in qrels.column_names else "corpus_id"
    qrel_rows = [{"query_id": str(r[q_col]), "doc_id": str(r[d_col]),
                  "score": int(r["score"])} for r in qrels]
    _dump(qrel_rows, cfg.QRELS_PATH)

    eval_qids = {r["query_id"] for r in qrel_rows}

    def title_text(r):
        t = (r.get("title") or "").strip()
        body = (r.get("text") or "").strip()
        return (t + ". " + body).strip(". ") if t else body

    _dump(({"_id": str(r["_id"]), "text": title_text(r)} for r in corpus),
          cfg.CORPUS_PATH)
    _dump(({"_id": str(r["_id"]), "text": r["text"]}
           for r in queries if str(r["_id"]) in eval_qids),
          cfg.QUERIES_PATH)

    print(f"dataset   : {cfg.DATASET}")
    print(f"corpus    : {len(corpus)} docs")
    print(f"queries   : {len(eval_qids)} (with qrels, split={cfg.QRELS_SPLIT})")
    print(f"qrels     : {len(qrel_rows)} judgements")
    print(f"written   : {cfg.CORPUS_PATH.name}, {cfg.QUERIES_PATH.name}, "
          f"{cfg.QRELS_PATH.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
