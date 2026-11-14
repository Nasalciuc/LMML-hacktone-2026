"""Final attribution with seven cameras: the five ordinary regulars, plus toorisugari split into its two cameras
("toorisugari" = camera T, "toorisugari_B" = camera B; see groups.py for the evidence).

Seeds: credited photos of the five regulars; for toorisugari the credited photos clearly on one side of the
spectral split. Every other photo (all anonymous + the remaining credited toorisugari ones) is assigned
iteratively (EM-style, z-scored leave-one-out correlation; a photo joins a fingerprint once its margin > 3).
Writes attribution.csv (anonymous photos) and attribution_credited_toorisugari.csv.
"""
import csv
from pathlib import Path

import numpy as np
import torch

import attribute2 as A

HERE = Path(__file__).resolve().parent
H7 = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin", "toorisugari_B"]
B_cred = "w01_012 w05_006 w05_013 w06_001".split()
T_cred = "w07_014 w02_012 w03_001 w04_011 w07_001 w02_018 w01_004 w04_009 w03_009".split()

names = A.names
N = A.N
seed = np.full(N, -1)
for i, n in enumerate(names):
    h, f = n.split("/")
    if h in ("amayadori", "jihanki", "rojiura", "shuuden", "yakin"):
        seed[i] = H7.index(h)
    elif h == "toorisugari":
        s = f[:-4]
        seed[i] = 4 if s in T_cred else (6 if s in B_cred else -1)
free = seed < 0
usable = A.M.flatten(1).float().mean(1).numpy() if hasattr(A, "M") else None


def sums(assign):
    num = torch.zeros(7, *A.W.shape[1:], device=A.dev)
    den = torch.zeros(7, *A.W.shape[1:], device=A.dev)
    for i in range(N):
        c = assign[i]
        if c < 0:
            continue
        wi = A.W[i].to(A.dev).float()
        li = A.I[i].to(A.dev).float()
        num[c] += wi * li
        den[c] += li * li
    return num, den


def score(i, num, den, assign):
    wi = A.W[i].to(A.dev).float()
    li = A.I[i].to(A.dev).float()
    ks = []
    for c in range(7):
        nu, de = num[c], den[c]
        if assign[i] == c:
            nu = nu - wi * li
            de = de - li * li
        ks.append(A.zm(nu / de.clamp(min=1)))
    K = torch.stack(ks)
    K = K - K.mean(0, keepdim=True)
    a = wi - wi.mean()
    b = K * li[None]
    dims = tuple(range(1, b.dim()))
    b = b - b.mean(dims, keepdim=True)
    na = a.norm()
    raw = (b * a).sum(dims) / (b.flatten(1).norm(dim=1) * na + 1e-12)
    nulls = torch.stack([(torch.roll(b, s, (-2, -1)) * a).sum(dims) / (b.flatten(1).norm(dim=1) * na + 1e-12)
                         for s in A.SHIFTS])
    sd = nulls.std(0).mean().clamp(min=1e-6)
    return (raw / sd).cpu().numpy()


def main():
    assign = seed.copy()
    for it in range(6):
        num, den = sums(assign)
        Z = np.stack([score(i, num, den, assign) for i in range(N)])
        srt = np.sort(Z, 1)
        margin = srt[:, -1] - srt[:, -2]
        best = Z.argmax(1)
        sd = ~free
        acc = (best[sd] == seed[sd]).mean()
        new = seed.copy()
        new[free & (margin > 3)] = best[free & (margin > 3)]
        print(f"iter {it}: seed LOO acc {acc:.3f}; free counts {np.bincount(best[free], minlength=7)}; "
              f"joined {int((free & (margin > 3)).sum())}", flush=True)
        if (new == assign).all():
            break
        assign = new
    np.save(HERE / "cache" / "Z7.npy", Z)
    with open(HERE / "attribution.csv", "w", newline="", encoding="utf-8") as fh, \
            open(HERE / "attribution_credited_toorisugari.csv", "w", newline="", encoding="utf-8") as fc:
        hdr = ["file", "photographer", "camera", "z_best", "z_margin"] + [f"z_{h}" for h in H7]
        wa = csv.writer(fh); wa.writerow(hdr)
        wc = csv.writer(fc); wc.writerow(hdr + ["seed"])
        for i in range(N):
            h, f = names[i].split("/")
            if h not in ("anonymous", "toorisugari"):
                continue
            z = Z[i]
            o = np.argsort(z)[::-1]
            cam = H7[o[0]]
            person = "toorisugari" if cam.startswith("toorisugari") else cam
            row = [f, person, cam, f"{z[o[0]]:.2f}", f"{z[o[0]] - z[o[1]]:.2f}"] + [f"{v:.2f}" for v in z]
            if h == "anonymous":
                wa.writerow(row)
            else:
                wc.writerow(row + [H7[seed[i]] if seed[i] >= 0 else ""])
    conf = np.zeros((7, 7), int)
    for i in np.where(sd)[0]:
        conf[seed[i], best[i]] += 1
    print("seed LOO confusion (rows = seed camera):\n", conf)


if __name__ == "__main__":
    main()
