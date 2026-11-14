"""Split the toorisugari attribution into its two sub-groups found by spectral analysis of the pairwise matrix,
compare group fingerprints with each other and with yakin's, and render the group fingerprints."""
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

A_anon = "w10_010 w10_018 w09_011 w08_023 w10_028 w12_002 w09_021 w08_002 w10_019 w08_006 w12_010 w09_014 w08_010 w12_001 w11_003".split()
A_cred = "w01_012 w05_006 w05_013 w06_001".split()
T_anon = "w08_004 w11_011 w09_022 w12_020 w11_028 w09_015 w12_008 w08_011 w11_016 w09_001 w11_029 w08_017 w11_014 w12_025 w11_012".split()
T_cred = "w07_014 w02_012 w03_001 w04_011 w07_001 w02_018 w01_004 w04_009 w03_009".split()

idx = lambda pref, lst: [names.index(f"{pref}/{s}.jpg") for s in lst]
groups = {
    "A_anon": idx("anonymous", A_anon), "A_cred": idx("toorisugari", A_cred),
    "T_anon": idx("anonymous", T_anon), "T_cred": idx("toorisugari", T_cred),
    "Y_cred": [i for i, n in enumerate(names) if n.startswith("yakin/")],
    "J_cred": [i for i, n in enumerate(names) if n.startswith("jihanki/")],
    "S_cred": [i for i, n in enumerate(names) if n.startswith("shuuden/")],
}


def fp(ix):
    num = sum(d["W"][i].float().to(dev) * d["I"][i].float().to(dev) for i in ix)
    den = sum(d["I"][i].float().to(dev) ** 2 for i in ix)
    k = FE.zm(num / den.clamp(min=1))
    return k, den


K = {g: fp(ix) for g, ix in groups.items()}
keys = list(K)


def corr(a, b):
    a = a.flatten(); b = b.flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


print("fingerprint correlations x1000 (independent photo sets)")
print("        " + " ".join(f"{k:>7s}" for k in keys))
for a in keys:
    print(f"{a:7s} " + " ".join(f"{corr(K[a][0], K[b][0]) * 1000:7.2f}" for b in keys))

if "--render" in sys.argv:
    for g in ("A_anon", "T_anon"):
        k, den = K[g]
        k = k.mean(0)  # average the colour channels
        for b in (4, 8):
            a = F.avg_pool2d(k[None, None], b, b)[0, 0]
            a = (a - a.mean()) / a.std()
            Image.fromarray(np.clip(a.cpu().numpy() * 50 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"group_{g}_b{b}.png")
        e = F.avg_pool2d((k * k)[None, None], 8, 8)[0, 0]
        e = (e - e.median()) / e.std()
        Image.fromarray(np.clip(e.cpu().numpy() * 50 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"group_{g}_energy8.png")
