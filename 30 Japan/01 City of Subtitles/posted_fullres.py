"""Full-resolution look for an additive pattern shared by toorisugari's anonymous photos in *posted*
coordinates. Earlier renders used block means of a high-pass residual, which cancel letter edges. Here:
mean residual of the 45 photos (posted coords) minus the mean of the 100 other anonymous photos, with
toorisugari's own sensor pattern (turned to each photo's orientation) regressed out; rendered with small
Gaussian blurs and as |gradient|. Output: out/pf_*.png"""
import csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
names = d["names"]
att = {r["file"]: r for r in csv.DictReader(open(HERE / "attribution_table.csv", encoding="utf-8"))}
S = torch.load(HERE / "cache" / "sensor_toorisugari.pt")
K = (S if torch.is_tensor(S) else S[list(S)[0]]).float()
print("sensor K", K.shape)
K = K.mean(0) if K.dim() == 3 else K
def acc(sel):
    num = torch.zeros(960, 1280); den = torch.zeros(960, 1280)
    for i in sel:
        w = d["W"][i].float().mean(0); m = d["M"][i].float()
        if m.dim() == 3: m = m[0]
        f = names[i].split("/")[1]
        if att[f]["photographer"] == "toorisugari":
            k = torch.flip(K, (0, 1)) if att[f]["rotated180"] == "1" else K
            I = d["I"][i].float().mean(0)
            kk = k * I * m
            a = float((w * kk).sum() / (kk * kk).sum().clamp(min=1e-9))
            w = w - a * kk
        num += w * m; den += m
    return (num / den.clamp(min=1)).numpy()
anon = [i for i, n in enumerate(names) if n.startswith("anonymous/")]
T = [i for i in anon if att[names[i].split("/")[1]]["photographer"] == "toorisugari"]
O = [i for i in anon if i not in T]
mT, mO = acc(T), acc(O)
np.save(HERE / "cache" / "pf_T.npy", mT); np.save(HERE / "cache" / "pf_O.npy", mO)
def save(a, name):
    a = a - np.median(a); s = np.percentile(np.abs(a), 99.5) + 1e-9
    Image.fromarray(np.clip(127 + 127 * a / s, 0, 255).astype(np.uint8)).save(HERE / "out" / name)
for tag, a in [("T", mT), ("O", mO), ("TminusO", mT - mO)]:
    for s in (1.5, 4):
        save(ndimage.gaussian_filter(a, s), f"pf_{tag}_g{s}.png")
    g = np.hypot(ndimage.sobel(ndimage.gaussian_filter(a, 1), 0), ndimage.sobel(ndimage.gaussian_filter(a, 1), 1))
    save(ndimage.gaussian_filter(g, 3), f"pf_{tag}_grad.png")
    e = ndimage.uniform_filter(a * a, 9)
    save(e, f"pf_{tag}_energy9.png")
print("done")
