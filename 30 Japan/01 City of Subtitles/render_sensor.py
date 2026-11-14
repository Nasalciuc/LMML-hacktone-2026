"""Render each photographer's sensor fingerprint (all photos, each turned back to sensor orientation),
SNR-normalised, at several smoothing scales: out/sensor_<handle>_b<k>.png and _energy16.png."""
import csv
import sys
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
att = {r["file"]: r for r in csv.DictReader(open(HERE / "attribution.csv"))}
cre = {(r["credited_as"], r["file"]): r for r in csv.DictReader(open(HERE / "credited_check.csv"))}
rot = lambda t: torch.flip(t, (-2, -1))


def info(i):
    h, f = names[i].split("/")
    if h == "anonymous":
        return att[f]["photographer"], int(att[f]["rotated180"])
    return h, int(cre[(h, f)]["rotated180"])


def sensor_fp(h, min_margin=0.0):
    num = 0
    den = 0
    n = 0
    for i in range(len(names)):
        hh, r = info(i)
        if hh != h:
            continue
        w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
        if r:
            w, li = rot(w), rot(li)
        num = num + w * li
        den = den + li * li
        n += 1
    return FE.zm(num / den.clamp(min=1)), den, n


def save(a, path):
    a = (a - a.median()) / (a.std() + 1e-12)
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(path)


if __name__ == "__main__":
    for h in sys.argv[1:]:
        k, den, n = sensor_fp(h)
        torch.save({"K": k.half().cpu(), "den": den.half().cpu()}, HERE / "cache" / f"sensor_{h}.pt")
        s = (k * den.sqrt()).mean(0)
        valid = (den.mean(0) > den.mean(0).median() * 0.2).float()
        s = s * valid
        for b in (2, 4, 8, 16):
            save(F.avg_pool2d(s[None, None], b, b)[0, 0], OUT / f"sensor_{h}_b{b}.png")
        save(F.avg_pool2d((s * s)[None, None], 16, 16)[0, 0], OUT / f"sensor_{h}_energy16.png")
        print(h, n, "photos")
