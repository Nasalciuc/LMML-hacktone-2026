"""Assemble model.pt for main.py from finished checkpoints.

Contents: the genuine-photo bank (128x128 RGB of every training photo, 32x32 search keys, byte hashes,
labels), the CNN ensemble (fp16 state dicts) and its thresholds.
python build_model.py --tag r128 --full_seeds 2 [--fallback_precision 0.9]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import shoelib as L

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "task_vietnam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--full_seeds", type=int, default=2)
    ap.add_argument("--fallback_precision", type=float, default=0.9)
    args = ap.parse_args()
    cv = json.load(open(HERE / "results" / f"{args.tag}_cv.json"))
    rec = cv["recipe"]
    oof = np.load(HERE / "results" / f"{args.tag}_oof.npz")
    # fallback threshold: smallest MSP whose accepted out-of-fold predictions reach the wanted precision
    thr = 1.01
    for t in np.unique(oof["msp"])[::-1]:
        a = oof["msp"] >= t
        if a.sum() >= 10 and (oof["pred"][a] == oof["y"][a]).mean() >= args.fallback_precision:
            thr = float(t)
    a = oof["msp"] >= thr
    print(f"CNN fallback threshold {thr:.3f}: accepts {a.mean():.3f} of OOF knowns, precision "
          f"{(oof['pred'][a] == oof['y'][a]).mean() if a.any() else float('nan'):.3f}")

    rows = list(csv.DictReader(open(DATA / "train-output.csv")))
    bank, keys, hashes, labels = [], [], [], []
    for r in rows:
        p = DATA / "train-input" / r["filename"]
        hashes.append(hashlib.sha1(p.read_bytes()).hexdigest())
        im = Image.open(p).convert("RGB")
        bank.append(np.asarray(im.resize((128, 128), Image.BILINEAR, reducing_gap=None), dtype=np.uint8))
        g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR, reducing_gap=None), dtype=np.float32).ravel()
        g = g - g.mean()
        keys.append(g / (np.linalg.norm(g) + 1e-6))
        labels.append(L.CLASSES.index(r["label"]))
    sds = []
    for s in range(args.full_seeds):
        sd = torch.load(HERE / "results" / "ckpt" / f"{args.tag}_full_s{s}.pt", map_location="cpu")
        sds.append({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()})
    ck = {"classes": L.CLASSES, "res": rec["res"], "width": rec["width"],
          "mean": L.MEAN.flatten().tolist(), "std": L.STD.flatten().tolist(),
          "models": sds, "cnn_threshold": thr,
          "bank": torch.from_numpy(np.stack(bank)), "keys": torch.from_numpy(np.stack(keys)),
          "hashes": hashes, "labels": torch.tensor(labels),
          "match_mean_max": 2.0, "match_block_max": 6.0, "recipe": rec}
    torch.save(ck, HERE / "model.pt")
    print("wrote model.pt", (HERE / "model.pt").stat().st_size // 1_000_000, "MB")


if __name__ == "__main__":
    main()
