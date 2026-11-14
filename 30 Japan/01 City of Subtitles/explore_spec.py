"""Where in the spectrum does the per-person pattern live? Cross-spectrum of split-half fingerprints."""
import numpy as np
import torch
import torch.nn.functional as F

import prnu

dev = "cuda"
credited, anonymous = prnu.list_photos()
handles = list(credited)


def lum_and_mask(p):
    rgb = prnu.load_rgb(p)
    m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
    x = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
    return x, m


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


halves = {}
for h in handles:
    hv = []
    for half in (0, 1):
        num = 0
        den = 0
        for p in credited[h][half::2]:
            x, m = lum_and_mask(p)
            y = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(3, 1, 1)).sum(0)[None, None]
            r = (y - wiener(y, 3))[0, 0] * m
            lv = boxf((r * r)[None, None], 15)[0, 0]
            r = zm(r / lv.clamp(min=1e-3).sqrt()) * m
            lum = y[0, 0] * m
            num = num + r * lum
            den = den + lum * lum
        hv.append(zm(num / den.clamp(min=1)))
    halves[h] = hv

H, W = halves[handles[0]][0].shape
fy = torch.fft.fftfreq(H, device=dev).view(-1, 1)
fx = torch.fft.fftfreq(W, device=dev).view(1, -1)
rad = torch.sqrt(fy ** 2 + fx ** 2)
bins = torch.linspace(0, 0.71, 15, device=dev)


def cross(a, b):
    A = torch.fft.fft2(a); B = torch.fft.fft2(b)
    c = (A * B.conj()).real
    pa = (A.abs() ** 2); pb = (B.abs() ** 2)
    out = []
    for i in range(len(bins) - 1):
        sel = (rad >= bins[i]) & (rad < bins[i + 1])
        out.append(float(c[sel].sum() / torch.sqrt(pa[sel].sum() * pb[sel].sum())))
    return np.array(out)


same = np.array([cross(*halves[h]) for h in handles])
diff = np.array([cross(halves[a][0], halves[b][1]) for a in handles for b in handles if a != b])
print("band  " + " ".join(f"{(bins[i]+bins[i+1]).item()/2:6.3f}" for i in range(len(bins) - 1)))
for h, s in zip(handles, same):
    print(f"{h[:6]:6s}" + " ".join(f"{v*1000:6.1f}" for v in s))
print("diffM " + " ".join(f"{v*1000:6.1f}" for v in diff.mean(0)))
print("diffS " + " ".join(f"{v*1000:6.1f}" for v in diff.std(0)))
# check separately the per-axis spectrum (periodic artefacts)
for h in handles[:2]:
    A = torch.fft.fft2(halves[h][0]); B = torch.fft.fft2(halves[h][1])
    c = (A * B.conj()).real
    top = torch.topk(c.flatten(), 10)
    print(h, [(int(i) // W, int(i) % W) for i in top.indices])
