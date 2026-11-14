"""Texture maps of a sensor fingerprint: local energy in several frequency bands and directions. Letters made
of differently textured noise (smoother, striped, ...) would show in some of these even if plain energy is flat."""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import prnu
from render_sensor import sensor_fp, OUT

h = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
K, den, _ = sensor_fp(h)
k = (K * den.sqrt()).mean(0)[None, None]
k = k / k.std()
lo = prnu.gblur(k, 1.0)
bands = {"low": lo, "high": k - lo,
         "dx": k[..., :, 1:] - k[..., :, :-1], "dy": k[..., 1:, :] - k[..., :-1, :],
         "checker": k * ((torch.arange(k.shape[-2], device=k.device).view(-1, 1) + torch.arange(k.shape[-1], device=k.device).view(1, -1)) % 2 * 2 - 1)}
tiles = []
for nm, b in bands.items():
    e = F.avg_pool2d(b * b, 16, 16)[0, 0]
    if nm == "checker":
        e = F.avg_pool2d(prnu.gblur(b, 1.0) ** 2, 16, 16)[0, 0]
    e = torch.log(e + 1e-6)
    a = (e - e.median()) / e.std()
    tiles.append(Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((640, 480), Image.NEAREST))
s = Image.new("L", (1920, 960))
for i, t in enumerate(tiles):
    s.paste(t, ((i % 3) * 640, (i // 3) * 480))
s.save(OUT / f"texture_{h}.png")
print("ok")
