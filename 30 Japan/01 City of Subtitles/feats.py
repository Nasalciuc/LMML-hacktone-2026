"""Residual extraction used by the final pipeline (GPU).

residual = per colour channel, image - db4 wavelet Wiener denoised image (sigma 3, 4 levels), divided by its
local RMS (15x15) so textured areas do not dominate,
row/column means removed, caption band and clipped pixels masked out.
Cached as cache/feats.pt (float16) for all 253 photos.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prnu

HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def boxf(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)


def wiener(x, k=3):
    mu = boxf(x, k)
    var = boxf(x * x, k) - mu * mu
    nv = var.mean(dim=(2, 3), keepdim=True)
    return mu + (var - nv).clamp(min=0) / var.clamp(min=1e-6) * (x - mu)


def zm(a):
    a = a - a.mean(-1, keepdim=True)
    return a - a.mean(-2, keepdim=True)


def one(path):
    rgb = prnu.load_rgb(path)
    m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
    x = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
    import wavelet
    y = x[None]  # 1,3,H,W
    r = wavelet.residual(y, 3.0)[0] * m  # 3,H,W
    lv = boxf((r * r)[None], 15)[0]
    r = zm(r / lv.clamp(min=1e-3).sqrt()) * m
    return r, y[0] * m, m


def load_all():
    f = HERE / "cache" / "feats.pt"
    credited, anonymous = prnu.list_photos()
    names = [f"{h}/{p.name}" for h in credited for p in credited[h]] + [f"anonymous/{p.name}" for p in anonymous]
    if f.exists():
        d = torch.load(f)
        return d
    paths = [p for h in credited for p in credited[h]] + list(anonymous)
    W, I, M = [], [], []
    for p in paths:
        r, lum, m = one(p)
        W.append(r.half().cpu()); I.append(lum.half().cpu()); M.append(m.bool().cpu())
    d = {"names": names, "W": torch.stack(W), "I": torch.stack(I), "M": torch.stack(M)}
    (HERE / "cache").mkdir(exist_ok=True)
    torch.save(d, f)
    return d


if __name__ == "__main__":
    d = load_all()
    print(len(d["names"]), d["W"].shape)
