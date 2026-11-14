"""Look for a second pattern in toorisugari's photos, in picture coordinates (as posted, not as shot).

1) toorisugari's sensor fingerprint K (all its photos, each turned back to sensor orientation).
2) for each of its anonymous photos, residual minus the fitted K term, in posted orientation.
3) MLE average of those cleaned residuals = candidate hidden pattern P; leave-one-out z of each photo
   (toorisugari's and everyone else's) against P tells whether P is real and whose photos carry it.
Saves cache/hidden_P.pt and renders out/hidden_*.png.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import feats as FE

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
dev = FE.dev
d = FE.load_all()
names = d["names"]
SHIFTS = [(97, 131), (211, 353), (-157, 419), (331, -277), (-283, -199), (149, 557), (-401, 89), (53, -613)]

att = {r["file"]: r for r in csv.DictReader(open(HERE / "attribution.csv"))}
cre = {(r["credited_as"], r["file"]): r for r in csv.DictReader(open(HERE / "credited_check.csv"))}


def info(i):
    h, f = names[i].split("/")
    if h == "anonymous":
        r = att[f]
        return r["photographer"], int(r["rotated180"])
    r = cre[(h, f)]
    return h, int(r["rotated180"])


rot = lambda t: torch.flip(t, (-2, -1))
toor = [i for i in range(len(names)) if info(i)[0] == "toorisugari"]
anon_t = [i for i in toor if names[i].startswith("anonymous")]

num = 0
den = 0
for i in toor:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    if info(i)[1]:
        w, li = rot(w), rot(li)
    num = num + w * li
    den = den + li * li
K = FE.zm(num / den.clamp(min=1))


def cleaned(i):
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    k = rot(K) if info(i)[1] else K
    s = k * li
    beta = (w * s).sum() / (s * s).sum().clamp(min=1e-9)
    return w - beta * s, li


pn = 0
pd = 0
cache = {}
for i in anon_t:
    w, li = cleaned(i)
    pn = pn + w * li
    pd = pd + li * li
P = FE.zm(pn / pd.clamp(min=1))
torch.save({"P": P.half().cpu(), "pd": pd.half().cpu()}, HERE / "cache" / "hidden_P.pt")


def z_against(i, P):
    w, li = cleaned(i) if info(i)[0] == "toorisugari" else (d["W"][i].float().to(dev), d["I"][i].float().to(dev))
    if i in anon_t:
        Pi = FE.zm((pn - w * li) / (pd - li * li).clamp(min=1))
    else:
        Pi = P
    a = w - w.mean()
    b = Pi * li
    b = b - b.mean()
    raw = float((a * b).sum() / (a.norm() * b.norm()))
    nulls = [float((a * torch.roll(b, s, (-2, -1))).sum() / (a.norm() * b.norm())) for s in SHIFTS]
    return raw / np.std(nulls)


rows = []
for i in range(len(names)):
    h, r = info(i)
    rows.append((names[i], h, r, z_against(i, P)))
for grp in ("toorisugari-anon", "toorisugari-credited", "others"):
    sel = [x for x in rows if (x[1] == "toorisugari") == grp.startswith("toor") and
           (grp == "others" or x[0].startswith("anonymous") == (grp == "toorisugari-anon"))]
    zs = np.array([x[3] for x in sel])
    print(f"{grp:22s} n={len(sel):3d} mean z {zs.mean():6.2f} sd {zs.std():5.2f} min {zs.min():6.2f} max {zs.max():6.2f}")
with open(HERE / "out" / "hidden_pattern_z.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["photo", "photographer", "rotated180", "z_vs_hidden"])
    for x in rows:
        wr.writerow([x[0], x[1], x[2], f"{x[3]:.2f}"])

Pm = (P * pd.sqrt()).mean(0)
for b in (1, 2, 4, 8, 16):
    a = F.avg_pool2d(Pm[None, None], b, b)[0, 0] if b > 1 else Pm
    a = (a - a.median()) / a.std()
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"hidden_P_b{b}.png")
e = F.avg_pool2d((Pm * Pm)[None, None], 16, 16)[0, 0]
e = (e - e.median()) / e.std()
Image.fromarray(np.clip(e.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / "hidden_P_energy16.png")
print("rendered")
