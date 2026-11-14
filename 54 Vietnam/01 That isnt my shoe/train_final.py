"""Train the final ensemble on all 735 photos and write model.pt for main.py.

The rejection threshold on the ensemble max-softmax is taken from the out-of-fold scores of the same recipe
(results/<cv_tag>_oof.npz, produced by validate.py with the same --seeds): the highest threshold that keeps
out-of-fold known accuracy (correct AND accepted) at or above --target.

python train_final.py --cv_tag cv3 --seeds 3 --res 128 --epochs 200 --oe 0.5 --mixup 0.3 --target 0.89
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import shoelib as L

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "task_vietnam"


def pick_threshold(oof, target):
    msp, pred, y = oof["msp"], oof["pred"], oof["y"]
    best = 0.0
    for t in np.unique(msp):
        acc = np.mean((pred == y) & (msp >= t))
        if acc >= target:
            best = max(best, float(t))
    acc = float(np.mean((pred == y) & (msp >= best)))
    rej_known = float(np.mean(msp < best))
    return best, acc, rej_known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--oe", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=0.89)
    ap.add_argument("--out", default=str(HERE / "model.pt"))
    args = ap.parse_args()

    oof = np.load(HERE / "results" / f"{args.cv_tag}_oof.npz")
    thr, acc, rej_known = pick_threshold(oof, args.target)
    print(f"threshold {thr:.4f}: OOF known accuracy {acc:.3f}, knowns rejected {rej_known:.3f}")

    rows = list(csv.DictReader(open(DATA / "train-output.csv")))
    names = [r["filename"] for r in rows]
    y = torch.tensor([L.CLASSES.index(r["label"]) for r in rows])
    x = L.load_images(DATA / "train-input", names)
    sds = []
    for s in range(args.seeds):
        m = L.train_model(x, y, 3, res=args.res, epochs=args.epochs, width=args.width, oe_weight=args.oe,
                          mixup=args.mixup, label_smooth=args.ls, seed=100 + s)
        sds.append({k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                    for k, v in m.state_dict().items()})
        print("trained model", s, flush=True)

    g = x.float().mean(1, keepdim=True) / 255.0
    t = F.interpolate(g, size=(32, 32), mode="area").flatten(1)
    t = F.normalize(t - t.mean(1, keepdim=True), dim=1)
    ck = {"classes": L.CLASSES, "res": args.res, "width": args.width,
          "mean": L.MEAN.flatten().tolist(), "std": L.STD.flatten().tolist(),
          "models": sds, "threshold": thr,
          "thumbs": t.half().float(), "thumb_labels": y.clone(), "dup_sim": 0.995,
          "recipe": vars(args), "oof_known_acc": acc, "oof_known_rejected": rej_known}
    torch.save(ck, args.out)
    json.dump({"threshold": thr, "oof_known_acc": acc, "oof_known_rejected": rej_known, "recipe": vars(args)},
              open(HERE / "results" / "final_model.json", "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
