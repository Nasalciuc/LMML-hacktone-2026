"""Detection power of the pairwise PHAT-PCE test: the same test on *uncleaned* luma residuals for pairs of photos
that really share a sensor (toorisugari x toorisugari, sensor-aligned transform) vs pairs from different
photographers. A hidden sensor as strong as a real one would score like the first group."""
import random
from pathlib import Path
import numpy as np, torch
from pce_pairs_lib import pce_batch
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
D = torch.load(HERE / "cache" / "clean_res.pt"); names, who, rot = D["names"], D["who"], D["rot"]
def res(i):
    w = d["W"][i].float().mean(0); m = d["M"][i].float(); r = w * m
    return (r / r[m > 0].std()).cuda()
def pce(i, j):
    a, b = res(i), res(j)
    if rot[i] != rot[j]: b = torch.flip(b, (-2, -1))
    X = torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b)); X = X / X.abs().clamp(min=1e-12)
    p, pk, dy, dx = pce_batch(torch.fft.irfft2(X, s=a.shape)[None]); return float(p[0]), int(dy[0]), int(dx[0])
random.seed(0)
for h in ["toorisugari", "yakin", "amayadori"]:
    ix = [i for i in range(len(names)) if who[i] == h]
    same = [pce(*random.sample(ix, 2)) for _ in range(15)]
    print(h, "same-sensor PCE:", sorted(round(s[0]) for s in same), " at zero shift:", sum(1 for s in same if s[1] == 0 and s[2] == 0), "/15")
ix = list(range(len(names)))
diff = []
while len(diff) < 15:
    i, j = random.sample(ix, 2)
    if who[i] != who[j]: diff.append(pce(i, j)[0])
print("different-sensor PCE:", sorted(round(x) for x in diff))
