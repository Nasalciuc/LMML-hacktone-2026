"""Corpus loading/chunking (shared with build_index.py) and GPU passage encoding.

    python encode_passages.py [--model .models/gte-small] [--pool mean] [--out _onnx/pv_gte.npy]

Encoding needs torch + transformers (run with the miniconda env, ~1 min on GPU).
The query side runs the same model through onnxruntime in main.py (int8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "task_cuba", "data")


def load_corpus(data: str = DATA):
    """Unique documents (exact duplicates grouped), their ids and a short title."""
    meta = [json.loads(l) for l in open(os.path.join(data, "corpus_meta.jsonl"), encoding="utf-8")]
    texts, ids, titles, groups = [], [], [], {}
    for m in meta:
        with open(os.path.join(data, "corpus", m["id"] + ".txt"), encoding="utf-8") as f:
            text = f.read()
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in groups:
            groups[h].append(m["id"])
            continue
        groups[h] = [m["id"]]
        ids.append(groups[h])
        texts.append(text)
        # book chapters: drop the "CHAPTER IV." numbering, keep the words
        titles.append(m["title"] if m["kind"] == "wiki_article"
                      else m["title"].split(".", 1)[-1] if "." in m["title"] else "")
    return texts, ids, titles, len(meta)


def chunk(words, size, step):
    if len(words) <= size:
        return [words]
    out = []
    for s in range(0, len(words), step):
        out.append(words[s:s + size])
        if s + size >= len(words):
            break
    return out


def passages(texts, size=200, step=100):
    p_doc, p_text = [], []
    for d, text in enumerate(texts):
        for c in chunk(text.split(), size, step):
            p_doc.append(d)
            p_text.append(" ".join(c))
    return np.array(p_doc), p_text


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--model", default=os.path.join(HERE, ".models", "gte-small"))
    ap.add_argument("--pool", default="mean", choices=["cls", "mean"])
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--maxtok", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(HERE, "_onnx", "pv_gte.npy"))
    args = ap.parse_args()

    texts, _, titles, _ = load_corpus(args.data)
    p_doc, p_text = passages(texts, args.size, args.step)
    dense = [titles[d] + " " + t if titles[d] else t for d, t in zip(p_doc, p_text)]
    print("passages", len(dense), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).float().cuda().eval()  # gte-small is stored fp16; run it fp32 like the ONNX export
    pv = np.zeros((len(dense), model.config.hidden_size), np.float32)
    with torch.no_grad():
        for i in range(0, len(dense), 128):
            e = tok(dense[i:i + 128], padding=True, truncation=True, max_length=args.maxtok,
                    return_tensors="pt").to("cuda")
            h = model(**e).last_hidden_state
            if args.pool == "cls":
                v = h[:, 0]
            else:
                m = e["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            pv[i:i + 128] = torch.nn.functional.normalize(v, dim=1).cpu().numpy()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, pv)
    print("saved", args.out, pv.shape)


if __name__ == "__main__":
    main()
