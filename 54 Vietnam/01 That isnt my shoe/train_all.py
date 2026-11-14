"""Resumable training of the whole recipe: 5-fold CV (S seeds per fold, out-of-fold scores) + S full-data models.
Every finished model is saved to results/ckpt/ right away; a rerun skips what is already there.
At the end writes results/<tag>_oof.npz, results/<tag>_cv.json and model.pt (via train_final logic).

python train_all.py --tag r128 --seeds 1 --full_seeds 2 --res 128 --epochs 200 --oe 0.5 --mixup 0.3 --target 0.89
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import shoelib as L
from train_final import pick_threshold
from validate import split_idx

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "task_vietnam"


def get_model(path, x, y, args, seed):
    if path.exists():
        m = L.ShoeNet(3, args.width).to("cuda").to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(path, map_location="cuda"))
        m.eval()
        return m
    m = L.train_model(x, torch.as_tensor(y), 3, res=args.res, epochs=args.epochs, width=args.width,
                      oe_weight=args.oe, mixup=args.mixup, label_smooth=args.ls, seed=seed)
    torch.save({k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                for k, v in m.state_dict().items()}, path)
    print("saved", path.name, flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--full_seeds", type=int, default=2)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--oe", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=0.89)
    args = ap.parse_args()
    ck = HERE / "results" / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(DATA / "train-output.csv")))
    names = [r["filename"] for r in rows]
    y = np.array([L.CLASSES.index(r["label"]) for r in rows])
    x = L.load_images(DATA / "train-input", names)
    folds = split_idx(y, 5, 123)

    probs = np.zeros((len(y), 3))
    for f in range(5):
        tr = np.where(folds != f)[0]
        te = np.where(folds == f)[0]
        p = 0
        for s in range(args.seeds):
            m = get_model(ck / f"{args.tag}_f{f}_s{s}.pt", x[tr], y[tr], args, seed=10 * f + s)
            lg, _ = L.infer(m, x[te], args.res)
            p = p + lg.softmax(1).cpu().numpy()
        probs[te] = p / args.seeds
    pred = probs.argmax(1)
    msp = probs.max(1)
    np.savez(HERE / "results" / f"{args.tag}_oof.npz", pred=pred, y=y, msp=msp, probs=probs)
    acc = float((pred == y).mean())
    per = {L.CLASSES[c]: float((pred[y == c] == c).mean()) for c in range(3)}
    thr, kacc, krej = pick_threshold({"msp": msp, "pred": pred, "y": y}, args.target)
    curve = {f"{q:.2f}": [round(float(np.quantile(msp, q)), 4),
                          round(float(np.mean((pred == y) & (msp >= np.quantile(msp, q)))), 4)]
             for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)}
    cv = {"recipe": vars(args), "oof_closed_acc": acc, "per_class": per, "threshold": thr,
          "oof_known_acc_at_threshold": kacc, "oof_known_rejected_at_threshold": krej,
          "reject_quantile_curve(thr,known_acc)": curve}
    json.dump(cv, open(HERE / "results" / f"{args.tag}_cv.json", "w"), indent=1)
    print(json.dumps(cv, indent=1), flush=True)

    sds = []
    for s in range(args.full_seeds):
        get_model(ck / f"{args.tag}_full_s{s}.pt", x, y, args, seed=100 + s)
        sds.append(torch.load(ck / f"{args.tag}_full_s{s}.pt", map_location="cpu"))
    g = x.float().mean(1, keepdim=True) / 255.0
    t = F.interpolate(g, size=(32, 32), mode="area").flatten(1)
    t = F.normalize(t - t.mean(1, keepdim=True), dim=1)
    torch.save({"classes": L.CLASSES, "res": args.res, "width": args.width,
                "mean": L.MEAN.flatten().tolist(), "std": L.STD.flatten().tolist(),
                "models": sds, "threshold": thr, "thumbs": t, "thumb_labels": torch.as_tensor(y),
                "dup_sim": 0.995, "recipe": vars(args)}, HERE / "model.pt")
    print("wrote model.pt", flush=True)


if __name__ == "__main__":
    main()
