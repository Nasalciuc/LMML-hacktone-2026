"""Render a camera fingerprint as an image, looking for structure (a drawn word) inside it.

For the photos assigned to a camera: K = MLE fingerprint, SNR-normalised by sqrt(sum I^2).
Outputs (out/fp_<camera>_*.png): blurred K (shape drawn as a level shift), local energy of K (shape drawn as
'noise only inside the letters'), and the half-split cross energy (same, without the noise bias).
usage: python render_fp.py <camera> [<camera> ...]   (camera as in attribution.csv 'camera' column)
"""
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
d = FE.load_all()
names = d["names"]
dev = FE.dev


def members(cam):
    ix = []
    for fn in ("attribution_7cam.csv", "attribution_credited_toorisugari.csv"):
        p = HERE / fn
        if not p.exists():
            continue
        pref = "anonymous" if fn == "attribution_7cam.csv" else "toorisugari"
        for r in csv.DictReader(open(p)):
            if r["camera"] == cam:
                ix.append(names.index(f"{pref}/{r['file']}"))
    if cam not in ("toorisugari", "toorisugari_B"):
        ix += [i for i, n in enumerate(names) if n.startswith(cam + "/")]
    return sorted(set(ix))


def fp(ix):
    num = sum(d["W"][i].float().to(dev) * d["I"][i].float().to(dev) for i in ix)
    den = sum(d["I"][i].float().to(dev) ** 2 for i in ix)
    k = FE.zm(num / den.clamp(min=1))
    return (k * den.sqrt()).mean(0), den.mean(0)  # SNR map, channel-averaged


def save(a, path):
    a = (a - a.median()) / (a.std() + 1e-12)
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(path)


def main():
  for cam in sys.argv[1:]:
      ix = members(cam)
      print(cam, len(ix))
      k, den = fp(ix)
      valid = (den > den.median() * 0.2).float()
      k = k * valid
      for b in (8, 16):
          save(F.avg_pool2d(k[None, None], b, b)[0, 0], OUT / f"fp_{cam}_blur{b}.png")
          save(F.avg_pool2d((k * k)[None, None], b, b)[0, 0], OUT / f"fp_{cam}_energy{b}.png")
      k1, _ = fp(ix[0::2])
      k2, _ = fp(ix[1::2])
      save(F.avg_pool2d((k1 * k2 * valid)[None, None], 16, 16)[0, 0], OUT / f"fp_{cam}_cross16.png")


if __name__ == "__main__":
    main()
