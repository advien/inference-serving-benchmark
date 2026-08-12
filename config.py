"""Central config for the Inference Serving Benchmark (retrieval + reranking).

Single source of truth for dataset, models, paths and benchmark knobs so the
serving variants stay comparable and the whole thing is reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
CORPUS_PATH = DATA / "corpus.jsonl"          # {_id, text}
QUERIES_PATH = DATA / "queries.jsonl"        # {_id, text}
QRELS_PATH = DATA / "relevance_labels" / "qrels.jsonl"   # {query_id, doc_id, score}

INDEX = ROOT / "index"
EMB_PATH = INDEX / "corpus_emb.npy"          # dense corpus embeddings
IDMAP_PATH = INDEX / "corpus_ids.json"       # row -> doc_id
HNSW_PATH = INDEX / "corpus.hnsw"            # approximate index

REPORTS = ROOT / "reports"

for _p in (DATA_RAW, DATA / "relevance_labels", INDEX, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Dataset -------------------------------------------------------------
# BEIR SciFact: small (~5.2k docs, 300 test queries) with qrels -> lets us
# report recall@k / MRR / nDCG honestly. Framed as generic knowledge search.
DATASET = "BeIR/scifact"
QRELS_DATASET = "BeIR/scifact-qrels"
QRELS_SPLIT = "test"

# --- Models --------------------------------------------------------------
RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"     # bi-encoder, 384-dim
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"        # baseline cross-encoder
RERANKER_SMALL = "cross-encoder/ms-marco-MiniLM-L-2-v2"  # faster, smaller reranker

EMB_DIM = 384
MAX_QUERY_LEN = 64
MAX_DOC_LEN = 160        # scifact abstracts; caps cross-encoder cost on CPU

# --- Retrieval / rerank knobs -------------------------------------------
TOP_K_RETRIEVE = 50      # rerank depth (candidates from the retriever)
TOP_K_RERANK = 10        # final list after reranking
EVAL_KS = (1, 5, 10)     # report metrics @ these cutoffs

# HNSW (approximate retrieval) params
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# --- Serving / benchmark -------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000
RERANK_BATCH = 32        # cross-encoder scoring batch size
# Server benchmark: pin to a FIXED small thread count so numbers are stable and
# comparable across variants, while still reflecting real multi-core serving
# (unlike the on-device single-thread frame of project 05).
BENCH_THREADS = 4
SEED = 42
