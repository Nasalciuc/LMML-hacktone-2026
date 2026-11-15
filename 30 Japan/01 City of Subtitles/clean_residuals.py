"""Step 1 of the hidden-seventh-sensor search. For every photo: luma residual (mean of the 3 channel residuals
in cache/feats.pt) minus its own photographer's sensor term, fitted by least squares. The sensor fingerprint is
leave-one-out (the photo itself excluded) and is turned to the photo's orientation. Output, in *posted*
coordinates: cache/clean_res.pt {names, who, rot, R (float16, unit std inside the caption mask)}."""
import csv
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
names = d["names"]
att = {r["file"]: r for r in csv.DictReader(open(HERE / "attribution_table.csv", encoding="utf-8"))}
cre = {(r["credited_as"], r["file"]): r for r in csv.DictReader(open(HERE / "credited_check.csv", encoding="utf-8"))}
who, rot = [], []
for n in names:
    h, f = n.split("/")
    if h == "anonymous":
        who.append(att[f]["photographer"]); rot.append(int(att[f]["rotated180"]))
    else:
        who.append(h); rot.append(int(cre[(h, f)]["rotated180"]))
flip = lambda t: torch.flip(t, (-2, -1))
def get(i):
    w = d["W"][i].float().mean(0); I = d["I"][i].float().mean(0); m = d["M"][i].float()
    return w, I, m
num, den = {}, {}
for i in range(len(names)):
    w, I, m = get(i)
    if rot[i]: w, I, m = flip(w), flip(I), flip(m)
    h = who[i]
    num[h] = num.get(h, 0) + w * I * m; den[h] = den.get(h, 0) + I * I * m
R = torch.empty(len(names), 960, 1280, dtype=torch.float16)
fit = []
for i in range(len(names)):
    w, I, m = get(i); h = who[i]
    ws, Is, ms = (flip(w), flip(I), flip(m)) if rot[i] else (w, I, m)
    K = (num[h] - ws * Is * ms) / (den[h] - Is * Is * ms).clamp(min=1)
    K = K - K.mean(1, keepdim=True); K = K - K.mean(0, keepdim=True)
    k = flip(K) if rot[i] else K
    s = k * I * m
    a = float((w * s).sum() / (s * s).sum().clamp(min=1e-9))
    r = (w - a * s) * m
    r = r - (r.sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)) * m
    r = r / r[m > 0].std().clamp(min=1e-9)
    R[i] = r.half(); fit.append(a)
torch.save({"names": names, "who": who, "rot": rot, "R": R, "fit": fit}, HERE / "cache" / "clean_res.pt")
print("saved", R.shape, "fit a: min %.2f median %.2f max %.2f" % (min(fit), sorted(fit)[len(fit)//2], max(fit)))
