"""Is there a pattern shared in *posted* coordinates? For each photographer, fingerprint from photos posted as
shot (P0) and from photos that were turned 180 degrees (P1). In posted coordinates the sensor pattern of P1 is
turned, so P0 and P1 only agree through something added after the turn (by the account, or hidden on purpose).
Prints corr(P0, P1) (posted alignment) and corr(P0, rot180(P1)) (sensor alignment) per photographer."""
import csv
from pathlib import Path

import torch

import feats as FE

HERE = Path(__file__).resolve().parent
dev = FE.dev
d = FE.load_all()
names = d["names"]
att = {r["file"]: r for r in csv.DictReader(open(HERE / "attribution.csv"))}
cre = {(r["credited_as"], r["file"]): r for r in csv.DictReader(open(HERE / "credited_check.csv"))}
rot = lambda t: torch.flip(t, (-2, -1))


def info(i):
    h, f = names[i].split("/")
    if h == "anonymous":
        return att[f]["photographer"], int(att[f]["rotated180"])
    return h, int(cre[(h, f)]["rotated180"])


def fp(ix):
    num = sum(d["W"][i].float().to(dev) * d["I"][i].float().to(dev) for i in ix)
    den = sum(d["I"][i].float().to(dev) ** 2 for i in ix)
    return FE.zm(num / den.clamp(min=1))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


out = {}
for h in ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]:
    for scope in ("all", "anonymous"):
        ix = [i for i in range(len(names)) if info(i)[0] == h and (scope == "all" or names[i].startswith("anonymous"))]
        i0 = [i for i in ix if info(i)[1] == 0]
        i1 = [i for i in ix if info(i)[1] == 1]
        if len(i0) < 2 or len(i1) < 2:
            continue
        P0, P1 = fp(i0), fp(i1)
        print(f"{h:12s} {scope:9s} n0={len(i0):2d} n1={len(i1):2d}  posted corr {corr(P0, P1)*1000:7.2f}  "
              f"sensor corr {corr(P0, rot(P1))*1000:7.2f}  (x1000)", flush=True)
        out[(h, scope)] = (P0.half().cpu(), P1.half().cpu())
torch.save(out, HERE / "cache" / "posted_P.pt")
