"""Final attribution: six cameras x two orientations (as shot / turned 180 degrees).

Finding behind it (orient_check.py): the "second toorisugari camera" is toorisugari's own sensor pattern turned
180 degrees (PCE 1542 at zero shift) - some photos were shot with the phone upside down and turned upright.
So every photo is scored against every fingerprint in both orientations; fingerprints are accumulated in the
sensor's own orientation. Credited photos seed the fingerprints (their orientation is estimated too).
Leave-one-out z-scores, EM refinement with anonymous photos whose margin > 3.

Writes attribution.csv: file, photographer, rotated (0/1), z_best, z_margin, z per (handle, orientation).
"""
import csv
from pathlib import Path

import numpy as np
import torch

import feats as FE

HERE = Path(__file__).resolve().parent
dev = FE.dev
SHIFTS = [(97, 131), (211, 353), (-157, 419), (331, -277), (-283, -199), (149, 557), (-401, 89), (53, -613),
          (389, 241), (-71, -463), (263, 29), (-337, 509)]

d = FE.load_all()
names = d["names"]
N = len(names)
H6 = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
person = np.array([H6.index(n.split("/")[0]) if not n.startswith("anonymous") else -1 for n in names])


def rot(t):
    return torch.flip(t, (-2, -1))


def get(i, r):
    w = d["W"][i].to(dev).float()
    li = d["I"][i].to(dev).float()
    return (rot(w), rot(li)) if r else (w, li)


def sums(assign, orient):
    num = torch.zeros(6, *d["W"].shape[1:], device=dev)
    den = torch.zeros(6, *d["W"].shape[1:], device=dev)
    for i in range(N):
        if assign[i] < 0:
            continue
        w, li = get(i, orient[i])
        num[assign[i]] += w * li
        den[assign[i]] += li * li
    return num, den


def score(i, num, den, assign, orient):
    """z for the 12 hypotheses (handle c, rotation r), index 2*c + r."""
    out = np.zeros(12)
    for r in (0, 1):
        w, li = get(i, r)
        ks = []
        for c in range(6):
            nu, de = num[c], den[c]
            if assign[i] == c:  # leave the photo out of its own fingerprint (it was added in its own orientation)
                w0, l0 = get(i, orient[i])
                nu = nu - w0 * l0
                de = de - l0 * l0
            ks.append(FE.zm(nu / de.clamp(min=1)))
        K = torch.stack(ks)
        K = K - K.mean(0, keepdim=True)
        a = w - w.mean()
        b = K * li[None]
        dims = tuple(range(1, b.dim()))
        b = b - b.mean(dims, keepdim=True)
        nb = b.flatten(1).norm(dim=1)
        raw = (b * a).sum(dims) / (nb * a.norm() + 1e-12)
        nulls = torch.stack([(torch.roll(b, s, (-2, -1)) * a).sum(dims) / (nb * a.norm() + 1e-12) for s in SHIFTS])
        z = raw / nulls.std(0).mean().clamp(min=1e-6)
        out[r::2] = z.cpu().numpy()
    return out


def main():
    assign = person.copy()
    orient = np.zeros(N, int)
    for it in range(7):
        num, den = sums(assign, orient)
        Z = np.stack([score(i, num, den, assign, orient) for i in range(N)])
        best = Z.argmax(1)
        srt = np.sort(Z, 1)
        margin = srt[:, -1] - srt[:, -2]
        cr = person >= 0
        acc = (best[cr] // 2 == person[cr]).mean()
        an = ~cr
        new_a = person.copy()
        new_o = np.zeros(N, int)
        new_o[cr] = Z[cr].reshape(-1, 6, 2)[np.arange(cr.sum()), person[cr]].argmax(1)
        sel = an & (margin > 3)
        new_a[sel] = best[sel] // 2
        new_o[sel] = best[sel] % 2
        print(f"iter {it}: credited LOO handle acc {acc:.3f}; anon counts {np.bincount(best[an] // 2, minlength=6)}; "
              f"anon joined {int(sel.sum())}; rotated credited {[int(new_o[person == c].sum()) for c in range(6)]}",
              flush=True)
        if (new_a == assign).all() and (new_o == orient).all():
            break
        assign, orient = new_a, new_o
    np.save(HERE / "cache" / "Z_final.npy", Z)
    hyp = [f"{h}{'_rot180' if r else ''}" for h in H6 for r in (0, 1)]
    with open(HERE / "attribution.csv", "w", newline="", encoding="utf-8") as fa, \
            open(HERE / "credited_check.csv", "w", newline="", encoding="utf-8") as fc:
        hdr = ["file", "photographer", "rotated180", "z_best", "z_margin"] + [f"z_{h}" for h in hyp]
        wa = csv.writer(fa); wa.writerow(hdr)
        wc = csv.writer(fc); wc.writerow(["credited_as"] + hdr)
        for i in range(N):
            z = Z[i]
            o = np.argsort(z)[::-1]
            row = [names[i].split("/")[1], H6[o[0] // 2], int(o[0] % 2), f"{z[o[0]]:.2f}", f"{z[o[0]] - z[o[1]]:.2f}"] + \
                  [f"{v:.2f}" for v in z]
            if person[i] < 0:
                wa.writerow(row)
            else:
                wc.writerow([H6[person[i]]] + row)
    conf = np.zeros((6, 6), int)
    for i in np.where(person >= 0)[0]:
        conf[person[i], best[i] // 2] += 1
    print("credited LOO confusion (rows true):\n", conf)


if __name__ == "__main__":
    main()
