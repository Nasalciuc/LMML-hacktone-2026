"""Submission network: EfficientNet-B0 (ImageNet weights) fine-tuned on the 735 task photos only.

python ko_final.py --tag r256_s10 [--res 256] [--epochs 40] [--smooth 0.1] [--loss ce]
-> results/ko_final_<tag>.pth  (fp32 state dict)
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch

import oe_lib as L

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--loss", default="ce", choices=["ce", "bce"]); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    L.RES = a.res
    t = np.load(HERE / "cache/train.npz")
    xk = torch.from_numpy(t["x"]).permute(0, 3, 1, 2).contiguous()
    yk = np.array([L.CLASSES.index(s) for s in t["y"]])
    t0 = time.time()
    m = L.train_model(xk, yk, xk[:0], arch=a.arch, epochs=a.epochs, lr=a.lr, smooth=a.smooth, seed=a.seed,
                      n_out=3, loss_kind=a.loss, log=lambda s: print(s, flush=True))
    out = HERE / "results" / f"ko_final_{a.tag}.pth"
    torch.save({k: v.float().cpu() for k, v in m.state_dict().items()}, out)
    print("saved", out, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
