"""'Same photo or doctored copy?' check against the 735 labelled photos.

Every labelled photo is kept as a 64x64 grey thumbnail. For a query we find the closest thumbnail
(mirror included) and measure  mean = mean |diff|  and  blk = worst 8x8-block mean |diff|  (0-255 grey).
  mean <= M_SAME and blk <= B_SAME      -> the labelled photo itself (re-saved/resized/mirrored): its label
  mean <= M_NEAR and blk >  B_EDIT      -> a known photo with a local change ("counterfeit"): unknown
  otherwise                             -> leave it to the network
This script simulates both cases on the training photos and prints the distance distributions
used to pick the thresholds.  python oe_bank.py
"""
import io, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
S = 64
M_SAME, B_SAME, M_NEAR, B_EDIT = 1.5, 3.5, 3.0, 5.0     # thresholds used in the exported model


def thumb(x01):
    """x01: float N,3,H,W in 0-1 -> N,64,64 grey 0-255 (float)."""
    g = (x01 * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    g = F.interpolate(g, size=[S, S], mode="bilinear", antialias=True, align_corners=False)
    return (g[:, 0] * 255.0)


def bank_from_u8(x_u8):
    return thumb(x_u8.float() / 255).round().clamp(0, 255).to(torch.uint8)


def match(q, bank):
    """q: N,64,64 float 0-255; bank: M,64,64 uint8 -> (best index, mean, blk) per query."""
    B = bank.float()
    best_i, best_m, best_b = [], [], []
    for t in [q, q.flip(2)]:
        d = (t.unsqueeze(1) - B.unsqueeze(0)).abs()            # N,M,64,64
        mean = d.mean((2, 3))
        i = mean.argmin(1)
        dd = d[torch.arange(len(q)), i]                          # N,64,64
        blk = F.avg_pool2d(dd.unsqueeze(1), 8).flatten(1).max(1).values
        best_i.append(i); best_m.append(mean.min(1).values); best_b.append(blk)
    sel = (best_m[1] < best_m[0]).long()
    pick = lambda a: torch.stack(a, 1)[torch.arange(len(q)), sel]
    return pick(best_i), pick(best_m), pick(best_b)


def pil_tf(a, kind, rng, others):
    im = Image.fromarray(a)
    if kind == "jpeg":
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(rng.integers(60, 96))); return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "resize":
        s = int(rng.integers(160, 400)); return np.asarray(im.resize((s, s), Image.BILINEAR).resize((256, 256), Image.BICUBIC))
    if kind == "mirror":
        return a[:, ::-1].copy()
    if kind == "png_same":
        return a.copy()
    # local edits ("counterfeit"): one 24-48 px square changed
    out = a.copy()
    h = int(rng.integers(24, 49)); w = int(rng.integers(24, 49))
    y = int(rng.integers(40, 216 - h)); x = int(rng.integers(40, 216 - w))
    patch = out[y:y + h, x:x + w]
    if kind == "paste":
        o = others[rng.integers(len(others))]; yy, xx = rng.integers(0, 256 - h), rng.integers(0, 256 - w)
        out[y:y + h, x:x + w] = o[yy:yy + h, xx:xx + w]
    elif kind == "blur":
        out[y:y + h, x:x + w] = np.asarray(Image.fromarray(patch).filter(ImageFilter.GaussianBlur(4)))
    elif kind == "recolour":
        out[y:y + h, x:x + w] = patch[..., ::-1]                # swap R/B
    elif kind == "flatten":
        out[y:y + h, x:x + w] = patch.reshape(-1, 3).mean(0).astype(np.uint8)
    elif kind == "shift":                                       # move a detail a few px
        dy, dx = rng.integers(-8, 9, 2); out[y:y + h, x:x + w] = a[y + dy:y + dy + h, x + dx:x + dx + w]
    return out


def main():
    t = np.load(HERE / "cache/train.npz"); x = t["x"]
    bank = bank_from_u8(torch.from_numpy(x).permute(0, 3, 1, 2))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), 150, replace=False)
    res = {}
    for kind in ["png_same", "jpeg", "resize", "mirror", "paste", "blur", "recolour", "flatten", "shift"]:
        q = np.stack([pil_tf(x[i], kind, rng, x) for i in idx])
        qi, qm, qb = match(thumb(torch.from_numpy(q).permute(0, 3, 1, 2).float() / 255), bank)
        ok = (qi.numpy() == idx).mean()
        same = ((qm <= M_SAME) & (qb <= B_SAME)).float().mean().item()
        edit = ((qm <= M_NEAR) & (qb > B_EDIT)).float().mean().item()
        res[kind] = dict(right_photo=float(ok), called_same=same, called_doctored=edit, mean_q50=float(qm.median()), mean_max=float(qm.max()),
                         blk_min=float(qb.min()), blk_q05=float(qb.quantile(0.05)), blk_q50=float(qb.median()), blk_max=float(qb.max()))
        print(f"{kind:9s} " + " ".join(f"{k} {v:.2f}" for k, v in res[kind].items()))
    # each training photo vs its closest *other* training photo
    B = bank.float(); m_other, b_other = [], []
    for i in range(len(B)):
        d = (B[i:i + 1] - B).abs(); mean = d.mean((1, 2)); mean[i] = 1e9
        j = mean.argmin(); m_other.append(mean[j].item())
        b_other.append(F.avg_pool2d(d[j][None, None], 8).max().item())
    m_other, b_other = np.array(m_other), np.array(b_other)
    res["other_photo"] = dict(mean_min=float(m_other.min()), mean_q01=float(np.quantile(m_other, 0.01)), mean_q50=float(np.median(m_other)),
                              blk_min=float(b_other.min()))
    print("closest other photo:", res["other_photo"])
    # new photos (3000 HF photos, never in the bank): how often would the check fire by accident?
    h = np.load(HERE / "cache/hf.npz"); hi = rng.choice(len(h["brand"]), 3000, replace=False)
    ms, bs = [], []
    for i in range(0, 3000, 50):
        _, m, b = match(thumb(torch.from_numpy(h["x"][hi[i:i + 50]]).permute(0, 3, 1, 2).float() / 255), bank)
        ms.append(m); bs.append(b)
    m, b = torch.cat(ms), torch.cat(bs)
    res["new_photos"] = dict(n=3000, mean_min=float(m.min()), mean_q01=float(m.quantile(0.01)),
                             called_same=float(((m <= M_SAME) & (b <= B_SAME)).float().mean()),
                             called_doctored=float(((m <= M_NEAR) & (b > B_EDIT)).float().mean()))
    print("new photos:", res["new_photos"])
    res["thresholds"] = dict(M_SAME=M_SAME, B_SAME=B_SAME, M_NEAR=M_NEAR, B_EDIT=B_EDIT)
    json.dump(res, open(HERE / "results/oe_bank_calib.json", "w"), indent=1)


if __name__ == "__main__":
    main()
