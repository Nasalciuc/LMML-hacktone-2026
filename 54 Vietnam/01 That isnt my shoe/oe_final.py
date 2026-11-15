"""Train the submission network on all 735 knowns + all outlier brands.

python oe_final.py --epochs 15 --out results/oe_final.pth
"""
import argparse, time

import numpy as np
import torch

import oe_lib as L
from oe_cv import load, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--u_ratio", type=float, default=1.0)
    ap.add_argument("--hf_known", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RES_DIR / "oe_final.pth"))
    ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args()
    L.RES = a.res
    xk, yk, xh, brand = load()
    unk = np.where(~np.isin(brand, list(L.NOT_UNKNOWN)))[0]
    if a.hf_known:
        kn = {"adidas": 0, "converse": 1, "nike": 2}
        hk = np.where(np.isin(brand, list(kn)))[0]
        hk = np.random.default_rng(0).permutation(hk)[: a.hf_known]
        xk = torch.cat([xk, xh[hk]]); yk = np.concatenate([yk, [kn[b] for b in brand[hk]]])
    print(f"knowns {len(xk)}, outliers {len(unk)}", flush=True)
    t0 = time.time()
    m = L.train_model(xk, yk, xh[unk], arch=a.arch, epochs=a.epochs, lr=a.lr, u_ratio=a.u_ratio, seed=a.seed, log=print)
    torch.save({k: v.float().cpu() for k, v in m.state_dict().items()}, a.out)
    print("saved", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
