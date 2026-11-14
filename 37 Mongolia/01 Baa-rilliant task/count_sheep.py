"""Sheep counting by density-map regression (small U-Net trained from scratch).

    python count_sheep.py --data ../task_mongolia --val 40     # hold out 40 train photos, report MAE
    python count_sheep.py --data ../task_mongolia --val 0      # train on all 280, predict the survey

Writes counts.csv (image,count) and model.pt.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SIGMA = 3.0      # gaussian per dot, in pixels of the 600x600 photo
SCALE = 100.0    # density is multiplied by this for the loss


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def dot_map(pts, h, w):
    m = np.zeros((h, w), np.float32)
    for x, y in pts:
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        m[yi, xi] += 1
    return m


def gauss_blur(t, sigma):
    r = int(3 * sigma)
    k = torch.exp(-(torch.arange(-r, r + 1, device=t.device) ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).float()
    t = F.conv2d(F.pad(t, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, r, r)), k.view(1, 1, -1, 1))


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(3, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.b = block(8 * c, 8 * c)
        self.d4, self.d3, self.d2, self.d1 = block(16 * c, 4 * c), block(8 * c, 2 * c), block(4 * c, c), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        up = lambda t, s: F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat([up(b, e4), e4], 1))
        d3 = self.d3(torch.cat([up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return F.softplus(self.out(d1))


def norm(x):  # uint8 NHWC -> float NCHW
    return (x.permute(0, 3, 1, 2).float() / 255.0 - 0.45) / 0.25


@torch.no_grad()
def predict(model, imgs, dev):
    model.eval()
    out = []
    for i in range(0, len(imgs), 8):
        x = norm(torch.from_numpy(np.stack(imgs[i:i + 8])).to(dev))
        d = 0
        for k in range(4):  # rot90 + flip test-time augmentation
            for fl in (False, True):
                xx = torch.rot90(x, k, (2, 3))
                xx = torch.flip(xx, (3,)) if fl else xx
                d = d + model(xx).sum((1, 2, 3))
        out += (d / 8 / SCALE).cpu().tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../task_mongolia"))
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("counts.csv"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"

    counts = pd.read_csv(a.data / "train_counts.csv")
    pts = pd.read_csv(a.data / "train_points.csv")
    # train_107.jpg shows ~75 sheep but is labelled 0 (no points): drop it
    counts = counts[~counts.image.isin(["train_107.jpg"])].reset_index(drop=True)
    names = list(counts.image)
    imgs = [load(a.data / "train" / n) for n in names]
    maps = [dot_map(pts[pts.image == n][["x", "y"]].values, *imgs[i].shape[:2]) for i, n in enumerate(names)]
    for i, n in enumerate(names):
        assert abs(maps[i].sum() - counts['count'][i]) < 1e-3, n

    perm = np.random.RandomState(1).permutation(len(names))
    va, tr = perm[:a.val], perm[a.val:]
    X = torch.from_numpy(np.stack([imgs[i] for i in tr])).to(dev)
    D = gauss_blur(torch.from_numpy(np.stack([maps[i] for i in tr]))[:, None].to(dev), SIGMA) * SCALE

    model = UNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.iters, pct_start=0.05)
    crop, bs = 320, 16
    for it in range(a.iters):
        model.train()
        idx = torch.randint(0, len(X), (bs,), device=dev)
        y0, x0 = np.random.randint(0, 600 - crop + 1, 2)
        x = norm(X[idx, y0:y0 + crop, x0:x0 + crop])
        d = D[idx, :, y0:y0 + crop, x0:x0 + crop]
        k = np.random.randint(4)
        x, d = torch.rot90(x, k, (2, 3)), torch.rot90(d, k, (2, 3))
        if np.random.rand() < 0.5:
            x, d = torch.flip(x, (3,)), torch.flip(d, (3,))
        # colour jitter per sample
        x = x * (1 + 0.2 * (torch.rand(bs, 3, 1, 1, device=dev) - 0.5)) + 0.3 * (torch.rand(bs, 1, 1, 1, device=dev) - 0.5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x.contiguous(memory_format=torch.channels_last))
        p = p.float()
        loss = F.mse_loss(p, d) + 0.01 * (p.sum((1, 2, 3)) - d.sum((1, 2, 3))).abs().mean() / SCALE
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 1000 == 0:
            msg = f"iter {it + 1} loss {loss.item():.4f}"
            if a.val:
                pv = predict(model, [imgs[i] for i in va], dev)
                gt = counts['count'].values[va]
                msg += f"  val MAE {np.abs(pv - gt).mean():.2f}  RMSE {np.sqrt(((pv - gt) ** 2).mean()):.2f}"
            print(msg, flush=True)

    if a.val:
        pv = predict(model, [imgs[i] for i in va], dev)
        gt = counts['count'].values[va]
        pd.DataFrame({"image": [names[i] for i in va], "true": gt, "pred": pv.round(2)}).to_csv("val_predictions.csv", index=False)
        print("validation MAE", np.abs(pv - gt).mean().round(2), "mean true", gt.mean().round(1))
        return

    torch.save(model.state_dict(), "model.pt")
    plots = pd.read_csv(a.data / "survey_plots.csv")
    simgs = [load(a.data / "survey" / n) for n in plots.image]
    pv = predict(model, simgs, dev)
    pd.DataFrame({"image": plots.image, "count": np.round(pv).astype(int)}).to_csv(a.out, index=False)
    pd.DataFrame({"image": plots.image, "count_raw": pv.round(2)}).to_csv("counts_raw.csv", index=False)
    print("wrote", a.out, "total sheep", int(np.round(pv).sum()))


if __name__ == "__main__":
    main()
