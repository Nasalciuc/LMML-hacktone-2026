"""Additive (intensity-independent) pattern shared in posted coordinates: plain mean residual of the photos
posted as shot vs of the photos turned 180 (their sensor patterns do not line up, a posted-frame pattern does).
Also the same split for random halves of all anonymous photos of other photographers, as a baseline."""
import torch

from render_sensor import info, d, names, dev


def mean_res(ix):
    s = sum(d["W"][i].float().to(dev) for i in ix)
    c = sum(d["M"][i].float().to(dev) for i in ix)
    return s / c.clamp(min=1)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


for h in ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]:
    for scope in ("anonymous", "credited"):
        ix = [i for i in range(len(names)) if info(i)[0] == h and names[i].startswith("anonymous") == (scope == "anonymous")]
        i0 = [i for i in ix if info(i)[1] == 0]; i1 = [i for i in ix if info(i)[1] == 1]
        if len(i0) < 2 or len(i1) < 2:
            continue
        print(f"{h:12s} {scope:9s} n0={len(i0):2d} n1={len(i1):2d} posted-frame corr x1000 {corr(mean_res(i0), mean_res(i1))*1000:6.2f}", flush=True)
