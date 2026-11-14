"""That isn't my shoe - inference.

python main.py [--input test-input] [--output predictions.csv] [--model model.pt]

Every image in test-input/ gets nike / adidas / converse or unknown, written to predictions.csv
(filename,label). Needs only torch, numpy and Pillow; GPU if present, else CPU.

Decision, in order:
1. Genuine: the photo is one of the labelled photos (same bytes, or the same pixels after a re-save /
   resize / mirror: mean difference <= 2 and no 8x8 block (at 128 px) off by more than 6 grey levels).
   It keeps that photo's label. A copy with a local change (the "counterfeit") fails the block test.
2. Optional (USE_CNN_FALLBACK, off by default): the CNN ensemble (trained from scratch) may name the brand when its
   max-softmax clears a strict threshold (95 % out-of-fold precision); everything else is unknown.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
USE_CNN_FALLBACK = False  # off: it accepts counterfeit near-copies of training photos (see README)


# ----------------------------------------------------------------------------- network (same as training)
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class ShoeNet(nn.Module):
    def __init__(self, n_classes=3, width=64, drop=0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                                  nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        chans = [w, 2 * w, 4 * w, 8 * w]
        layers = []
        cin = w
        for i, c in enumerate(chans):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
            cin = c
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(self.drop(F.adaptive_avg_pool2d(x, 1).flatten(1)))


# ----------------------------------------------------------------------------- genuine-photo test
def small(im, s):
    return np.asarray(im.resize((s, s), Image.BILINEAR, reducing_gap=None), dtype=np.float32)


def key(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR, reducing_gap=None), dtype=np.float32).ravel()
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def diff(a, b):
    d = np.abs(a - b).mean(2)
    s = d.shape[0]
    return float(d.mean()), float(d.reshape(s // 8, 8, s // 8, 8).mean((1, 3)).max())


def genuine_label(path, im, ck, hash_index, keys, keys_flip):
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    if h in hash_index:
        return int(ck["labels"][hash_index[h]])
    k = key(im)
    sim = np.maximum(keys @ k, keys_flip @ k)
    a = small(im, 128)
    for j in np.argsort(-sim)[:5]:
        b = ck["bank"][j].numpy().astype(np.float32)
        m, blk = min(diff(a, b), diff(a, b[:, ::-1]), key=lambda t: t[1])
        if m <= ck["match_mean_max"] and blk <= ck["match_block_max"]:
            return int(ck["labels"][j])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test-input")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--model", default=str(HERE / "model.pt"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    hash_index = {h: i for i, h in enumerate(ck["hashes"])}
    keys = ck["keys"].numpy()
    keys_flip = keys.reshape(-1, 32, 32)[:, :, ::-1].reshape(len(keys), -1)
    models = []
    for sd in ck["models"]:
        m = ShoeNet(len(classes), ck["width"])
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        models.append(m.to(dev).eval())
    mean = torch.tensor(ck["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(ck["std"], device=dev).view(1, 3, 1, 1)

    folder = Path(args.input)
    names = sorted(f for f in os.listdir(folder) if Path(f).suffix.lower() in EXTS)
    rows = []
    how = {"genuine": 0, "cnn": 0, "unknown": 0}
    with torch.no_grad():
        for n in names:
            im = Image.open(folder / n).convert("RGB")
            lab = genuine_label(folder / n, im, ck, hash_index, keys, keys_flip)
            if lab is not None:
                rows.append((n, classes[lab]))
                how["genuine"] += 1
                continue
            label = "unknown"
            if USE_CNN_FALLBACK and models:
                x = torch.from_numpy(np.asarray(im.resize((256, 256), Image.BILINEAR), dtype=np.float32).copy())
                x = x.permute(2, 0, 1)[None].to(dev) / 255.0
                x = F.interpolate(x, size=(ck["res"], ck["res"]), mode="bilinear", antialias=True, align_corners=False)
                x = (x - mean) / std
                p = sum((m(x).softmax(1) + m(torch.flip(x, dims=[3])).softmax(1)) / 2 for m in models) / len(models)
                msp, k = p.max(1)
                if float(msp) >= ck["cnn_threshold"]:
                    label = classes[int(k)]
                    how["cnn"] += 1
            if label == "unknown":
                how["unknown"] += 1
            rows.append((n, label))

    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "label"])
        w.writerows(rows)
    print(f"wrote {args.output}: {len(rows)} rows", how)


if __name__ == "__main__":
    main()
