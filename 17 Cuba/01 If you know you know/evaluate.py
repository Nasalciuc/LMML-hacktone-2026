"""Hit@3 on the public questions (two halves) and on extra_questions.jsonl.

    python evaluate.py [index_dir]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from main import Retriever

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "task_cuba", "data")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_sets():
    qs = read(os.path.join(DATA, "sample-input.jsonl"))
    gold = {r["qid"]: r["doc_id"] for r in read(os.path.join(DATA, "sample-output.jsonl"))}
    pub = [(q["qid"], q["question"], gold[q["qid"]]) for q in qs]
    extra = [(q["qid"], q["question"], q["doc_id"]) for q in read(os.path.join(HERE, "extra_questions.jsonl"))]
    return {"public_a": pub[0::2], "public_b": pub[1::2], "extra": extra}


def main():
    r = Retriever(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "index"))
    for name, rows in load_sets().items():
        hits, lat, miss = 0, [], []
        for qid, q, g in rows:
            t = time.perf_counter()
            ids = r.search(q, 3)
            lat.append((time.perf_counter() - t) * 1000)
            if g in ids:
                hits += 1
            else:
                miss.append(qid)
        print(f"{name:<9} n={len(rows):<4} hit@3={hits / len(rows):.3f}  p95={np.percentile(lat, 95):.1f} ms")
    print("misses (extra):", " ".join(miss))


if __name__ == "__main__":
    main()
