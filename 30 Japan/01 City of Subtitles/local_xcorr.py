"""Local (block-wise) correlation between two camera fingerprints. A word drawn by making one fingerprint copy
(or negate) the other only inside the letters would show up here even if the global correlation is ~0.
usage: python local_xcorr.py block camA camB [camC ...]   (camA against each of the others)"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from render_fp import members, fp, OUT

blk = int(sys.argv[1])
a = sys.argv[2]
ka, da = fp(members(a))
for b in sys.argv[3:]:
  kb, db = fp(members(b))
  valid = ((da > da.median() * 0.2) & (db > db.median() * 0.2)).float()
  num = F.avg_pool2d((ka * kb * valid)[None, None], blk, blk)[0, 0]
  ea = F.avg_pool2d((ka * ka * valid)[None, None], blk, blk)[0, 0]
  eb = F.avg_pool2d((kb * kb * valid)[None, None], blk, blk)[0, 0]
  r = num / (ea * eb).sqrt().clamp(min=1e-12)
  print(a, b, "global", float((ka * kb * valid).sum() / ((ka * ka * valid).sum() * (kb * kb * valid).sum()).sqrt()),
        "local mean", float(r.mean()), "sd", float(r.std()))
  img = np.clip((r / r.std()).cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)
  Image.fromarray(img).resize((1280, 960), Image.NEAREST).save(OUT / f"xcorr_{a}_{b}_{blk}.png")
