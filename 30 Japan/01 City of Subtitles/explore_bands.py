"""Split-half reliability of per-person fingerprints for different residual filters (GPU).
For each person: fingerprint from odd photos vs even photos. Same-person NCC should beat cross-person."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

import prnu

dev = "cuda"
credited, anonymous = prnu.list_photos()
handles = list(credited)


def load_all():
    xs, ms = [], []
    for h in handles:
        for p in credited[h]:
            rgb = prnu.load_rgb(p)
            xs.append(torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8))
            ms.append(torch.from_numpy(prnu.caption_mask(rgb)))
    return torch.stack(xs), torch.stack(ms)


def gblur(x, s):
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def resid(x, kind):
    if kind == "g1":
        return x - gblur(x, 1.0)
    if kind == "g05":
        return x - gblur(x, 0.6)
    if kind == "dog13":
        return gblur(x, 0.7) - gblur(x, 2.0)
    if kind == "dog36":
        return gblur(x, 2.0) - gblur(x, 5.0)
    if kind == "dog612":
        return gblur(x, 5.0) - gblur(x, 12.0)
    if kind == "lap":
        k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=x.device, dtype=torch.float32) / 4
        c = x.shape[1]
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c)
    raise ValueError


def zm(a):
    a = a - a.mean(-1, keepdim=True)
    return a - a.mean(-2, keepdim=True)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


X, M = load_all()
n_per = 18
for kind in sys.argv[1:] or ["g05", "g1", "lap", "dog13", "dog36", "dog612"]:
    for chan in ["luma", "R", "G", "B"]:
        fps = {}
        for hi, h in enumerate(handles):
            halves = []
            for half in (0, 1):
                num = 0; den = 0
                for j in range(half, n_per, 2):
                    i = hi * n_per + j
                    x = X[i:i + 1].to(dev).float()
                    if chan == "luma":
                        x = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
                    else:
                        x = x[:, "RGB".index(chan):"RGB".index(chan) + 1]
                    m = M[i].to(dev).float()
                    r = resid(x, kind)[0, 0]
                    lum = x[0, 0]
                    num = num + zm(r * m) * lum * m
                    den = den + (lum * m) ** 2
                halves.append(zm(num / den.clamp(min=1)))
            fps[h] = halves
        same = [ncc(fps[h][0], fps[h][1]) for h in handles]
        cross = [ncc(fps[a][0], fps[b][1]) for a in handles for b in handles if a != b]
        print(f"{kind:7s} {chan:4s} same " + " ".join(f"{s*1000:6.2f}" for s in same) +
              f" | cross mean {np.mean(cross)*1000:6.2f} sd {np.std(cross)*1000:5.2f} max {np.max(cross)*1000:6.2f}", flush=True)
