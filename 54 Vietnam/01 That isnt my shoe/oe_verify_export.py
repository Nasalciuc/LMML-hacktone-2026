"""Does the packed file decide like the fp32 checkpoint it was made from?
python oe_verify_export.py results/oe_final.pth vietnam_model.pt --res 256   (CPU)"""
import argparse

import numpy as np
import torch

import oe_lib as L
from oe_cv import load

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("pt"); ap.add_argument("--res", type=int, default=224)
ap.add_argument("--arch", default="efficientnet_b0")
a = ap.parse_args()
L.RES = a.res
torch.set_num_threads(8)
xk, yk, xh, brand = load()
rng = np.random.default_rng(0)
ih = rng.choice(len(xh), 1500, replace=False)
net = L.build(a.arch); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")); net.eval()
pt = torch.jit.load(a.pt, map_location="cpu")
for name, x in [("task photos (735)", xk), ("web photos (1500)", xh[ih])]:
    ref, new = [], []
    with torch.no_grad():
        for i in range(0, len(x), 100):
            xb = L.to_input(x[i:i + 100])
            ref.append((net(xb) + net(xb.flip(3))) / 2); new.append(pt.logits(x[i:i + 100]))
    ref, new = torch.cat(ref), torch.cat(new)
    dr, dn = L.decide(ref, pt.tau, pt.bias), L.decide(new, pt.tau, pt.bias)
    print(f"{name}: max |logit diff| {(ref-new).abs().max():.3f}, mean {(ref-new).abs().mean():.4f}, same decision {(dr == dn).float().mean():.4f}")
