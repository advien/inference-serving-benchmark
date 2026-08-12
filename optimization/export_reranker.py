"""Export the cross-encoder reranker to ONNX + dynamic int8.

Reuses the 05 optimization idea: the reranker is the per-query hotspot (it scores
TOP_K_RETRIEVE pairs), so an int8 ONNX reranker is the natural serving optimization.

Writes:
  artifacts/reranker/model.onnx        fp32 graph
  artifacts/reranker/model_int8.onnx   dynamic int8
  artifacts/reranker/<tokenizer files>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

OUT_DIR = cfg.ROOT / "artifacts" / "reranker"
OPSET = 14


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.RERANKER)
    model.eval()
    tok.save_pretrained(OUT_DIR)

    enc = tok([["what is a foo", "a foo is a bar"]], padding="max_length",
              truncation=True, max_length=cfg.MAX_DOC_LEN, return_tensors="pt")
    inputs = (enc["input_ids"], enc["attention_mask"],
              enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
    dyn = {0: "batch", 1: "seq"}
    fp32 = OUT_DIR / "model.onnx"

    torch.onnx.export(
        model, inputs, str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    int8 = OUT_DIR / "model_int8.onnx"
    quantize_dynamic(model_input=str(fp32), model_output=str(int8),
                     weight_type=QuantType.QInt8)

    mb = lambda p: round(p.stat().st_size / 1024 / 1024, 2)
    print(f"reranker ONNX fp32 -> {fp32.name} ({mb(fp32)} MB)")
    print(f"reranker ONNX int8 -> {int8.name} ({mb(int8)} MB)")


if __name__ == "__main__":
    main()
