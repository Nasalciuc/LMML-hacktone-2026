"""Builds index/ for main.py.

    python build_index.py [--pv _onnx/pv_gte.npy] [--onnx _onnx/gte_int8_pc.onnx] [--pca 256] [--out index]

Needs numpy, tokenizers, PyStemmer, onnxruntime (the venv). Run export_onnx.py and encode_passages.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from encode_passages import load_corpus, passages
from main import Retriever, analyse, normalise

HERE = os.path.dirname(os.path.abspath(__file__))
# gte-small takes the bare query; bge-small and arctic-embed need this prefix (pass it with --prefix, with --pool cls)
BGE_ARCTIC_PREFIX = "Represent this sentence for searching relevant passages: "


def postings(token_lists, dtype_tf):
    vocab: dict[str, int] = {}
    rows, cols, tfs = [], [], []
    for i, toks in enumerate(token_lists):
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            rows.append(vocab.setdefault(t, len(vocab)))
            cols.append(i)
            tfs.append(n)
    rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs)
    order = np.lexsort((cols, rows))
    rows, cols, tfs = rows[order], cols[order], tfs[order]
    indptr = np.zeros(len(vocab) + 1, np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    lens = np.array([len(t) for t in token_lists])
    tfs = np.minimum(tfs, np.iinfo(dtype_tf).max).astype(dtype_tf)
    return vocab, indptr.astype(np.int32), cols, tfs, lens


def gaps(idx, indptr):
    """Posting lists as gaps (first entry absolute), which zip compresses well."""
    out = idx.astype(np.int64).copy()
    out[1:] -= idx[:-1]
    starts = indptr[:-1][np.diff(indptr) > 0]
    out[starts] = idx[starts]
    return out.astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "task_cuba", "data"))
    ap.add_argument("--pv", default=os.path.join(HERE, "_onnx", "pv_gte.npy"))
    ap.add_argument("--onnx", default=os.path.join(HERE, "_onnx", "gte_int8_pc.onnx"))
    ap.add_argument("--tokenizer", default=os.path.join(HERE, ".models", "gte-small", "tokenizer.json"))
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--pca", type=int, default=256, help="passage vector dims kept (0 = no projection)")
    ap.add_argument("--weights", default='{"p": 0.4, "d": 0.1, "e": 0.5, "ed": 0.2}')
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "index"))
    args = ap.parse_args()

    texts, ids, titles, n_meta = load_corpus(args.data)
    print("documents", n_meta, "unique", len(texts))
    p_doc, p_text = passages(texts, args.size, args.step)
    print("passages", len(p_doc))

    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # BM25 postings for passages and whole documents
    p_toks = [analyse(titles[d] + " " + t) for d, t in zip(p_doc, p_text)]
    d_toks = [analyse(titles[d] + " " + t) for d, t in enumerate(texts)]
    vocab_p, ip, ix, tf, plen = postings(p_toks, np.uint8)
    vocab_d, dip, dix, dtf, dlen = postings(d_toks, np.uint16)
    terms = sorted(set(vocab_p) | set(vocab_d))
    tid = {t: i for i, t in enumerate(terms)}

    def remap(vocab, indptr):
        new = np.zeros(len(terms) + 1, np.int64)
        old_ids = np.array([vocab[t] for t in terms if t in vocab])
        new_ids = np.array([tid[t] for t in terms if t in vocab])
        new[new_ids + 1] = indptr[old_ids + 1] - indptr[old_ids]
        return np.cumsum(new).astype(np.int32), np.concatenate([np.arange(indptr[o], indptr[o + 1]) for o in old_ids])

    ip2, perm = remap(vocab_p, ip)
    dip2, dperm = remap(vocab_d, dip)
    assert len(p_doc) < 65536
    np.savez_compressed(os.path.join(out, "lex.npz"),
                        p_indptr=ip2, p_idx=gaps(ix[perm], ip2), p_tf=tf[perm], p_len=plen.astype(np.uint16),
                        d_indptr=dip2, d_idx=gaps(dix[dperm], dip2), d_tf=dtf[dperm], d_len=dlen.astype(np.int32),
                        p_doc=p_doc.astype(np.uint16))
    with open(os.path.join(out, "terms.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(terms))
    counts: dict[str, int] = {}
    for text in texts:
        for w in normalise(text).split():
            counts[w] = counts.get(w, 0) + 1
    with open(os.path.join(out, "words.txt"), "w", encoding="utf-8") as f:
        f.write("".join(f"{w}\t{n}\n" for w, n in sorted(counts.items()) if not w.isdigit()))

    # Dense: passage vectors from encode_passages.py, PCA-reduced, stored int8 with one scale per row
    pv = np.load(args.pv).astype(np.float32)
    assert len(pv) == len(p_doc), f"passage vectors {len(pv)} != passages {len(p_doc)}; re-run encode_passages.py"
    dense = {}
    if args.pca and args.pca < pv.shape[1]:
        # uncentred PCA (SVD on every 3rd passage): keeps the directions that carry the cosine
        _, _, vt = np.linalg.svd(pv[::3].astype(np.float64), full_matrices=False)
        dense["proj"] = np.ascontiguousarray(vt[:args.pca].T.astype(np.float32))
        pv = pv @ dense["proj"]
        pv /= np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9
    scale = np.abs(pv).max(axis=1, keepdims=True) / 127.0 + 1e-12
    np.savez(os.path.join(out, "dense.npz"), pv=np.round(pv / scale).astype(np.int8),
             pv_scale=scale.astype(np.float32).ravel(), **dense)
    shutil.copy(args.onnx, os.path.join(out, "encoder.onnx"))
    shutil.copy(args.tokenizer, os.path.join(out, "qtokenizer.json"))

    with open(os.path.join(out, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"size": args.size, "step": args.step, "model": os.path.basename(os.path.dirname(args.tokenizer)),
                   "pool": args.pool, "query_prefix": args.prefix, "k1": 1.2, "b": 0.75, "spell": True,
                   "dense_raw": False, "weights": json.loads(args.weights)}, f)

    total = sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out))
    print("index MB", round(total / 1e6, 2))
    Retriever(out)


if __name__ == "__main__":
    main()
