"""Builds index/ for main.py from the task corpus.

    python build_index.py [--data ../task_cuba/data] [--model .models/potion-retrieval-32M] [--out index]

Needs numpy, tokenizers, safetensors and PyStemmer (the same tokeniser code as main.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from main import analyse, normalise, Retriever

HERE = os.path.dirname(os.path.abspath(__file__))


def chunk(words, size, step):
    if len(words) <= size:
        return [words]
    out = []
    for s in range(0, len(words), step):
        out.append(words[s:s + size])
        if s + size >= len(words):
            break
    return out


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
    ap.add_argument("--model", default=os.path.join(HERE, ".models", "potion-retrieval-32M"))
    ap.add_argument("--out", default=os.path.join(HERE, "index"))
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--min-word", type=int, default=1)
    ap.add_argument("--idf-alpha", type=float, default=0.5)
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(os.path.join(args.data, "corpus_meta.jsonl"), encoding="utf-8")]
    texts, ids, titles, groups = [], [], [], {}
    for m in meta:
        with open(os.path.join(args.data, "corpus", m["id"] + ".txt"), encoding="utf-8") as f:
            text = f.read()
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in groups:
            groups[h].append(m["id"])
            continue
        groups[h] = [m["id"]]
        ids.append(groups[h])
        texts.append(text)
        titles.append(m["title"] if m["kind"] == "wiki_article" else m["title"].split(".", 1)[-1] if "." in m["title"] else "")
    print("documents", len(meta), "unique", len(texts))

    p_doc, p_text = [], []
    for d, text in enumerate(texts):
        for c in chunk(text.split(), args.size, args.step):
            p_doc.append(d)
            p_text.append(" ".join(c))
    p_doc = np.array(p_doc)
    print("passages", len(p_doc))

    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

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
        f.write("".join(f"{w}\t{n}\n" for w, n in sorted(counts.items()) if n >= args.min_word and not w.isdigit()))

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    emb = load_file(os.path.join(args.model, "model.safetensors"))
    emb = next(iter(emb.values())).astype(np.float32)[:, :args.dim]
    unk = tok.token_to_id("[UNK]")

    def encode(batch):
        encs = tok.encode_batch(batch, add_special_tokens=False)
        res = np.zeros((len(batch), emb.shape[1]), np.float32)
        for i, e in enumerate(encs):
            t = [x for x in e.ids if x != unk]
            if t:
                res[i] = emb[t].mean(axis=0)
        return res / (np.linalg.norm(res, axis=1, keepdims=True) + 1e-9)

    if args.idf_alpha:
        df = np.zeros(len(emb))
        for i in range(0, len(p_text), 2048):
            for e in tok.encode_batch(p_text[i:i + 2048], add_special_tokens=False):
                df[np.unique(e.ids)] += 1
        emb *= (np.log((len(p_text) + 1) / (df + 1)) ** args.idf_alpha)[:, None].astype(np.float32)

    pv = np.concatenate([encode(p_text[i:i + 2048]) for i in range(0, len(p_text), 2048)])
    dv = np.zeros((len(texts), pv.shape[1]), np.float32)
    np.add.at(dv, p_doc, pv)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9

    def q8(m):
        s = np.abs(m).max(axis=1, keepdims=True) / 127.0 + 1e-12
        return np.round(m / s).astype(np.int8), s.astype(np.float32).ravel()

    ev, es = q8(emb)
    pq, ps = q8(pv)
    np.savez(os.path.join(out, "dense.npz"), emb=ev, emb_scale=es, pv=pq, pv_scale=ps,
             dv=dv.astype(np.float16))
    shutil.copy(os.path.join(args.model, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    with open(os.path.join(out, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"size": args.size, "step": args.step, "dim": args.dim, "model": os.path.basename(args.model),
                   "k1": 1.2, "b": 0.75, "weights": {"p": 0.4, "d": 0.2, "e": 0.2, "ed": 0.2, "pe": 0.4}}, f)

    total = sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out))
    print("index MB", round(total / 1e6, 2))
    Retriever(out)


if __name__ == "__main__":
    main()
