"""Exports the query encoder to ONNX and quantizes it to int8 (per-channel).

    python export_onnx.py [--model .models/gte-small] [--out _onnx/gte_int8_pc.onnx]

Needs torch, transformers, onnx and onnxruntime (miniconda env). Per-channel int8 keeps the
query vectors at cosine ~0.998 to fp32; per-tensor int8 drops to ~0.97-0.99 and costs accuracy.
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import torch
    import torch.nn as nn
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, ".models", "gte-small"))
    ap.add_argument("--out", default=os.path.join(HERE, "_onnx", "gte_int8_pc.onnx"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModel.from_pretrained(args.model).float().eval()  # some checkpoints (gte-small) are stored fp16

    class Wrap(nn.Module):
        # transformers 5.x clashes with positional export args, so pass keywords explicitly
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    e = tok(["hello world"], return_tensors="pt")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fp32 = args.out.replace(".onnx", "_fp32.onnx")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(Wrap(base).eval(), tuple(e[n] for n in names), fp32,
                      input_names=names, output_names=["last_hidden_state"],
                      dynamic_axes={n: {0: "b", 1: "s"} for n in names + ["last_hidden_state"]},
                      opset_version=14, do_constant_folding=True, dynamo=False)
    quantize_dynamic(fp32, args.out, weight_type=QuantType.QInt8, per_channel=True)
    print("int8 model", round(os.path.getsize(args.out) / 1e6, 2), "MB ->", args.out)


if __name__ == "__main__":
    main()
