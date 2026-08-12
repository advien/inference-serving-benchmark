**Quality pick: `exact_onnx_rerank`** — nDCG@10 0.6541, recall@10 0.7987, p50 909.04 ms, 4328.44 cpu-ms/query.

**Cost/latency pick: `retrieve_only_exact`** — nDCG@10 0.6479 (within 0.02 of best), p50 10.71 ms (84.9x faster than the quality pick), 46.56 cpu-ms/query.