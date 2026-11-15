"""Local validation for a submission folder (main.py + index/).

    python evaluate.py [submission_dir] [--cv]

Reports hit@3 with the shipped weights on:
  public      the 268 sample questions (also split in two halves)
  extra       59 own questions on documents the public set never asks about
  wiki / non-wiki  public questions split by the kind of the gold document. The misses are all on
              books and reports, and the hidden set seems to be mostly those, so non-wiki is the number to watch
  hard        public questions that BM25 alone misses (the paraphrased "riddle" kind)
  typos       public questions with ~1 in 5 longer words garbled (swap/drop/double a letter)
plus p95 latency of search() on one thread. --cv also tunes the fusion weights with
3x repeated 5-fold CV on the public set and reports the held-out hit@3 (all, and non-wiki only),
which says how much of the public score is weight-fitting.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "task_cuba", "data")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def garble(q: str, rng: random.Random) -> str:
    out = []
    for w in q.split():
        if len(w) > 4 and w.isalpha() and rng.random() < 0.2:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(("swap", "drop", "double"))
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def load_retriever(sub: str):
    spec = importlib.util.spec_from_file_location("sub_main", os.path.join(sub, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, sub)
    spec.loader.exec_module(mod)
    cwd = os.getcwd()
    os.chdir(sub)
    t = time.perf_counter()
    r = mod.Retriever("index")
    load = time.perf_counter() - t
    os.chdir(cwd)
    return r, load


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sub = os.path.abspath(args[0] if args else HERE)
    r, load = load_retriever(sub)
    id2doc = {d: i for i, grp in enumerate(r.docs) for d in grp}

    qs = read(os.path.join(DATA, "sample-input.jsonl"))
    gold = {x["qid"]: x["doc_id"] for x in read(os.path.join(DATA, "sample-output.jsonl"))}
    pub = [(q["question"], gold[q["qid"]]) for q in qs]
    kinds = {m["id"]: m["kind"] for m in read(os.path.join(DATA, "corpus_meta.jsonl"))}
    nonwiki = np.array([kinds[g] != "wiki_article" for _, g in pub])
    extra = [(q["question"], q["doc_id"]) for q in read(os.path.join(HERE, "extra_questions.jsonl"))]
    rng = random.Random(7)
    typos = [(garble(q, rng), g) for q, g in pub]

    for q, _ in pub[:10]:  # warm-up, like the platform
        r.search(q, 3)

    def run(rows):
        hits, lat = [], []
        for q, g in rows:
            t = time.perf_counter()
            ids = r.search(q, 3)
            lat.append((time.perf_counter() - t) * 1000)
            hits.append(g in ids)
        return np.array(hits), lat

    h_pub, lat = run(pub)
    h_ext, _ = run(extra)
    h_typo, _ = run(typos)

    # BM25-only ranking decides which public questions are "hard"
    lex_w = {k: (v if k in ("p", "d") else 0.0) for k, v in r.w.items()}
    comps = [r.components(q) for q, _ in pub]
    bm25_hit = np.array([id2doc[g] in np.argsort(-r.fuse(c, lex_w))[:3] for c, (_, g) in zip(comps, pub)])
    hard = ~bm25_hit

    print(f"submission: {sub}")
    print(f"load {load:.2f} s   p95 {np.percentile(lat, 95):.1f} ms   max {max(lat):.1f} ms")
    print(f"public  n={len(pub):<4} hit@3 {h_pub.mean():.3f}   halves {h_pub[0::2].mean():.3f} / {h_pub[1::2].mean():.3f}")
    print(f"wiki    n={(~nonwiki).sum():<4} hit@3 {h_pub[~nonwiki].mean():.3f}")
    print(f"nonwiki n={nonwiki.sum():<4} hit@3 {h_pub[nonwiki].mean():.3f}   (books, reports, OCR)")
    print(f"extra   n={len(extra):<4} hit@3 {h_ext.mean():.3f}")
    print(f"hard    n={hard.sum():<4} hit@3 {h_pub[hard].mean():.3f}   (BM25 alone misses these)")
    print(f"typos   n={len(typos):<4} hit@3 {h_typo.mean():.3f}")

    if "--cv" in sys.argv:
        keys = [k for k in r.w]
        gold_i = np.array([id2doc[g] for _, g in pub])
        N = {k: np.array([r._norm(c[k]) for c in comps]) for k in keys}
        grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        def hit(w, sel):
            s = sum(w[i] * N[k][sel] for i, k in enumerate(keys))
            top = np.argpartition(-s, 3, axis=1)[:, :3]
            return np.mean([gold_i[sel][j] in top[j] for j in range(len(sel))])

        def tune(sel):
            w = [r.w[k] for k in keys]
            best = hit(w, sel)
            for _ in range(4):
                better = False
                for i in range(len(keys)):
                    for v in grid:
                        w2 = w[:]
                        w2[i] = v
                        h = hit(w2, sel)
                        if h > best + 1e-9:
                            best, w, better = h, w2, True
                if not better:
                    break
            return w

        scores, scores_nw = [], []
        for seed in range(3):
            folds = np.array_split(np.random.RandomState(seed).permutation(len(pub)), 5)
            for f in range(5):
                train = np.concatenate([folds[j] for j in range(5) if j != f])
                w = tune(train)
                scores.append(hit(w, folds[f]))
                scores_nw.append(hit(w, folds[f][nonwiki[folds[f]]]))
        print(f"cv      held-out hit@3 {np.mean(scores):.3f} +- {np.std(scores):.3f} (3x5 folds, weights tuned on train folds)")
        print(f"cv      held-out non-wiki hit@3 {np.mean(scores_nw):.3f} +- {np.std(scores_nw):.3f}")
        print(f"        weights tuned on all public: {dict(zip(keys, tune(np.arange(len(pub)))))}")


if __name__ == "__main__":
    main()
