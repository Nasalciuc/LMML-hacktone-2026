"""Posted-frame average of the raw noise residual (no whitening) for toorisugari's anonymous photos and for
everyone else's: a small faint mark at a fixed spot (a signature) would add up here. Saves full-size PNGs."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prnu
import wavelet

HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = list(csv.DictReader(open(HERE / "attribution_table.csv")))
groups = {"toorisugari": [r["file"] for r in rows if r["photographer"] == "toorisugari"],
          "others": [r["file"] for r in rows if r["photographer"] != "toorisugari"]}
for g, files in groups.items():
    acc = 0
    cnt = 0
    for f in files:
        rgb = prnu.load_rgb(prnu.DATA / "anonymous" / f)
        m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
        r = wavelet.residual(x, 3.0)[0].mean(0)
        r = r.clamp(-15, 15)
        acc = acc + r * m
        cnt = cnt + m
    a = acc / cnt.clamp(min=1)
    a = prnu.gblur(a[None, None], 1.0)[0, 0]
    v = (a / a.std()).cpu().numpy()
    np.save(HERE / "cache" / f"sig_{g}.npy", v.astype(np.float32))
    Image.fromarray(np.clip(v * 40 + 128, 0, 255).astype(np.uint8)).save(HERE / "out" / f"signature_{g}.png")
    print(g, len(files))
