"""Block-wise correlation between toorisugari's sensor fingerprint and every other photographer's (all photos,
each turned back to sensor orientation). Same phone model could mean shared pattern; a word written by
copying / negating another camera's pattern inside letters would draw itself here.
usage: python sensor_xcorr.py block [handle_a]"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from render_sensor import sensor_fp, OUT

blk = int(sys.argv[1]) if len(sys.argv) > 1 else 32
a = sys.argv[2] if len(sys.argv) > 2 else "toorisugari"
ka, da, _ = sensor_fp(a)
ka = ka * da.sqrt()
for b in ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]:
    if b == a:
        continue
    kb, db, _ = sensor_fp(b)
    kb = kb * db.sqrt()
    for nm, kk in (("same", kb), ("rot180", torch.flip(kb, (-2, -1)))):
        num = F.avg_pool2d((ka * kk).sum(0)[None, None], blk, blk)[0, 0]
        ea = F.avg_pool2d((ka * ka).sum(0)[None, None], blk, blk)[0, 0]
        eb = F.avg_pool2d((kk * kk).sum(0)[None, None], blk, blk)[0, 0]
        r = num / (ea * eb).sqrt().clamp(min=1e-12)
        g = float((ka * kk).sum() / ((ka * ka).sum() * (kk * kk).sum()).sqrt())
        print(f"{a} vs {b:11s} {nm:6s} global {g*1000:6.2f}  local mean {float(r.mean())*1000:6.2f} sd {float(r.std())*1000:6.2f} "
              f"|r|>3sd blocks {int((r.abs() > 3 * r.std()).sum())}", flush=True)
        v = (r - r.mean()) / r.std()
        Image.fromarray(np.clip(v.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"sx_{a}_{b}_{nm}_{blk}.png")
