"""Look for a hidden pattern shared by one photographer's anonymous photos.

For a group of photos it writes out/<tag>_*.png:
  lowpass  : average of band-passed luma (x - blur30), scene averages out, a faint drawn mark does not
  pairs    : block-wise sum over photo pairs of residual products (2*sum_{i<j} r_i r_j), with the person's own
             PRNU term projected out; a hidden noise texture shared by the group lights up
  meanres  : block-averaged mean residual
usage: python secret_scan.py <handle> [min_margin]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import feats as FE
import prnu

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
dev = FE.dev


def norm_img(a, path, scale=40, size=None):
    a = a.float()
    a = (a - a.median()) / (a.std() + 1e-12)
    im = Image.fromarray(np.clip(a.cpu().numpy() * scale + 128, 0, 255).astype(np.uint8))
    if size:
        im = im.resize(size, Image.NEAREST)
    im.save(path)


def group(handle, min_margin):
    rows = list(csv.DictReader(open(HERE / "attribution.csv")))
    return sorted(r["file"] for r in rows if r["photographer"] == handle and float(r["z_margin"]) >= min_margin)


def main():
    handle = sys.argv[1]
    mm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    files = group(handle, mm)
    print(handle, len(files), files)
    d = FE.load_all()
    names = d["names"]
    idx = [names.index("anonymous/" + f) for f in files]
    cred = [i for i, n in enumerate(names) if n.startswith(handle + "/")]

    # lowpass average of the photos themselves
    acc = 0
    cnt = 0
    for f in files:
        rgb = prnu.load_rgb(prnu.DATA / "anonymous" / f)
        m = torch.from_numpy(prnu.caption_mask(rgb)).float().to(dev)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev).mean(1, keepdim=True)
        bp = (prnu.gblur(x, 2.0) - prnu.gblur(x, 30.0))[0, 0]
        bp = bp / (bp[m > 0].std() + 1e-6)
        acc = acc + bp * m
        cnt = cnt + m
    norm_img(acc / cnt.clamp(min=1), OUT / f"{handle}_lowpass.png", 30, (640, 480))

    # pairwise residual products
    W = d["W"]
    I = d["I"]
    num = sum(W[i].float().to(dev) * I[i].float().to(dev) for i in cred)
    den = sum(I[i].float().to(dev) ** 2 for i in cred)
    K = FE.zm(num / den.clamp(min=1))
    S = 0
    SS = 0
    for i in idx:
        r = W[i].float().to(dev)
        kl = K * I[i].float().to(dev)
        beta = (r * kl).sum() / (kl * kl).sum().clamp(min=1e-9)
        r = r - beta * kl
        S = S + r
        SS = SS + r * r
    P = (S * S - SS).sum(0) if S.dim() == 3 else (S * S - SS)
    for b in (16, 32):
        pb = F.avg_pool2d(P[None, None], b, b)[0, 0]
        norm_img(pb, OUT / f"{handle}_pairs_b{b}.png", 30, (640, 480))
    Sm = S.mean(0) if S.dim() == 3 else S
    norm_img(F.avg_pool2d(Sm[None, None], 4, 4)[0, 0], OUT / f"{handle}_meanres_b4.png", 40, (640, 480))
    print("written to", OUT)


if __name__ == "__main__":
    main()
