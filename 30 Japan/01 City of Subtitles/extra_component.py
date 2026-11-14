"""Do a photographer's anonymous photos share something their credited photos do not? (sensor orientation)
Compare corr(anon half 1, anon half 2) with corr(anon half, credited). Equal sample sizes, so the two numbers
should match if all photos carry the same single pattern."""
import torch

from render_sensor import info, d, names, dev, rot, FE


def fp(ix):
    num = 0
    den = 0
    for i in ix:
        w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
        if info(i)[1]:
            w, li = rot(w), rot(li)
        num = num + w * li
        den = den + li * li
    return FE.zm(num / den.clamp(min=1))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


for h in ["toorisugari", "jihanki", "rojiura", "yakin", "shuuden", "amayadori"]:
    an = sorted(i for i in range(len(names)) if info(i)[0] == h and names[i].startswith("anonymous"))
    cr = [i for i in range(len(names)) if names[i].startswith(h + "/")]
    n = min(len(cr), len(an) // 2)
    a1, a2 = an[0::2][:n], an[1::2][:n]
    A1, A2, C = fp(a1), fp(a2), fp(cr[:n])
    print(f"{h:12s} n={n}  anon1-anon2 {corr(A1, A2)*1000:6.2f}  anon1-cred {corr(A1, C)*1000:6.2f}  anon2-cred {corr(A2, C)*1000:6.2f}", flush=True)
