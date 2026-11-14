"""CenterNet-style glyph detector trained from scratch on the labelled rubbings.

One heatmap per glyph id (210 ids + 'rare'), plus a box-size head. A glyph = a local maximum
of its class heatmap above a threshold.

    python detector.py train --data ../task_china --val 150 --epochs 80     # holdout run
    python detector.py train --data ../task_china --val 0   --epochs 80     # final model
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

STRIDE = 4
BG = 235.0


def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sk = None if (i == o and s == 1) else nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sk is None else self.sk(x)))


class Net(nn.Module):
    def __init__(self, ncls, w=(32, 64, 128, 256, 384)):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, w[0], 2), conv_bn(w[0], w[0]))           # /2
        self.l1 = nn.Sequential(Res(w[0], w[1], 2), Res(w[1], w[1]))                   # /4
        self.l2 = nn.Sequential(Res(w[1], w[2], 2), Res(w[2], w[2]), Res(w[2], w[2]))  # /8
        self.l3 = nn.Sequential(Res(w[2], w[3], 2), Res(w[3], w[3]), Res(w[3], w[3]))  # /16
        self.l4 = nn.Sequential(Res(w[3], w[4], 2), Res(w[4], w[4]), Res(w[4], w[4]))  # /32
        self.u3 = conv_bn(w[4] + w[3], w[3]); self.u2 = conv_bn(w[3] + w[2], w[2]); self.u1 = conv_bn(w[2] + w[1], 128)
        self.hm = nn.Sequential(conv_bn(128, 128), nn.Conv2d(128, ncls, 1))
        self.wh = nn.Sequential(conv_bn(128, 64), nn.Conv2d(64, 2, 1))
        self.hm[-1].bias.data.fill_(-4.6)

    def forward(self, x):
        x = self.stem(x); c1 = self.l1(x); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.u3(torch.cat([up(c4, c3), c3], 1))
        d2 = self.u2(torch.cat([up(d3, c2), c2], 1))
        d1 = self.u1(torch.cat([up(d2, c1), c1], 1))
        return self.hm(d1), self.wh(d1)


def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)


def norm(x):
    return (x - 128.0) / 64.0


def gaussian_targets(boxes, cls, H, W, ncls):
    """boxes in input pixels (x, y, w, h); returns heatmap (ncls, H/4, W/4), wh target, mask."""
    h4, w4 = H // STRIDE, W // STRIDE
    hm = np.zeros((ncls, h4, w4), np.float32)
    wh = np.zeros((2, h4, w4), np.float32)
    msk = np.zeros((h4, w4), np.float32)
    yy, xx = np.mgrid[0:h4, 0:w4]
    for (x, y, w, h), c in zip(boxes, cls):
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        if not (0 <= cx < w4 and 0 <= cy < h4):
            continue
        sx, sy = max(w / STRIDE / 6, 0.8), max(h / STRIDE / 6, 0.8)
        g = np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
        hm[c] = np.maximum(hm[c], g)
        ix, iy = int(cx), int(cy)
        hm[c, iy, ix] = 1.0
        wh[:, iy, ix] = (math.log(max(w, 2)), math.log(max(h, 2)))
        msk[iy, ix] = 1
    return hm, wh, msk


def focal(pred, gt):
    p = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pl = torch.log(p) * (1 - p) ** 2 * pos
    nl = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pl.sum() + nl.sum()) / n


class Data:
    def __init__(self, root, labels, names, classes):
        self.root, self.classes = root, {c: i for i, c in enumerate(classes)}
        self.imgs = {n: load_gray(root / "train" / "images" / n) for n in names}
        self.boxes = {n: g[["x", "y", "w", "h"]].values for n, g in labels[labels.image.isin(names)].groupby("image")}
        self.cls = {n: [self.classes[c] for c in g.glyph_id] for n, g in labels[labels.image.isin(names)].groupby("image")}
        self.names = names

    def sample(self, rng, size=512):
        n = self.names[rng.integers(len(self.names))]
        img = self.imgs[n]
        s = rng.uniform(0.7, 1.3)
        H, W = img.shape
        im = np.asarray(Image.fromarray(img).resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR), np.float32)
        b = self.boxes[n] * s
        # place in a size x size canvas at a random offset (crop or pad)
        H, W = im.shape
        canvas = np.full((size, size), BG + rng.normal(0, 8), np.float32)
        ox = rng.integers(min(0, size - W), max(0, size - W) + 1)
        oy = rng.integers(min(0, size - H), max(0, size - H) + 1)
        x0, y0 = max(0, ox), max(0, oy)
        sx0, sy0 = max(0, -ox), max(0, -oy)
        w_, h_ = min(W - sx0, size - x0), min(H - sy0, size - y0)
        canvas[y0:y0 + h_, x0:x0 + w_] = im[sy0:sy0 + h_, sx0:sx0 + w_]
        b = b.copy(); b[:, 0] += ox; b[:, 1] += oy
        # photometric
        canvas = (canvas - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-25, 25)
        if rng.random() < 0.3:
            canvas = canvas + rng.normal(0, rng.uniform(2, 12), canvas.shape)
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            canvas = np.asarray(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).filter(
                __import__("PIL.ImageFilter").ImageFilter.GaussianBlur(k / 3)), np.float32)
        keep = []
        for i, (x, y, w, h) in enumerate(b):
            cx, cy = x + w / 2, y + h / 2
            keep.append(0 <= cx < size and 0 <= cy < size)
        keep = np.array(keep)
        return canvas, b[keep], [c for c, k in zip(self.cls[n], keep) if k]


@torch.no_grad()
def detect(model, img, ncls_ids, thr=0.3, tta=True):
    """img: float gray HxW. Returns list of (cls, score, cx, cy, w, h) in image pixels."""
    H, W = img.shape
    Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    pad = np.full((Hp, Wp), BG, np.float32); pad[:H, :W] = img
    x = torch.from_numpy(norm(pad))[None, None].cuda()
    hm, wh = model(x)
    hm = hm.float().sigmoid()
    hm = hm[:, ncls_ids] if ncls_ids is not None else hm
    mx = F.max_pool2d(hm, 5, 1, 2)
    peaks = (hm == mx) & (hm >= thr)
    out = []
    for c, y, xx in peaks[0].nonzero().tolist():
        s = hm[0, c, y, xx].item()
        w, h = wh[0, :, y, xx].float().exp().tolist()
        out.append((c, s, (xx + 0.5) * STRIDE, (y + 0.5) * STRIDE, w, h))
    return out, hm[0, :, :H // STRIDE, :W // STRIDE].cpu().numpy()


def train(a):
    labels = pd.read_csv(a.data / "train_labels.csv")
    classes = sorted(labels.glyph_id.unique())
    names = sorted(labels.image.unique())
    rng = np.random.default_rng(a.seed)
    perm = np.random.RandomState(0).permutation(len(names))
    va = [names[i] for i in perm[:a.val]]
    tr = [names[i] for i in perm[a.val:]]
    data = Data(a.data, labels, tr, classes)
    model = Net(len(classes)).cuda().to(memory_format=torch.channels_last)
    steps = a.epochs * len(tr) // a.bs
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05)
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(8)
    t0 = time.time()

    def make_batch(seed):
        r = np.random.default_rng(seed)
        X, HM, WH, M = [], [], [], []
        for _ in range(a.bs):
            im, b, c = data.sample(r)
            hm, wh, m = gaussian_targets(b, c, 512, 512, len(classes))
            X.append(norm(im).astype(np.float32)); HM.append(hm); WH.append(wh); M.append(m)
        return [torch.from_numpy(np.stack(v)) for v in (X, HM, WH, M)]

    start = 0
    ckpt = a.out.with_suffix(".ckpt")
    if ckpt.exists():  # resume after an interruption
        ck = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"]
        rng = np.random.default_rng(a.seed + start)
        print("resumed at step", start, flush=True)
    fut = pool.submit(make_batch, rng.integers(1 << 30))
    for it in range(start, steps):
        X, HM, WH, M = fut.result()
        fut = pool.submit(make_batch, rng.integers(1 << 30))
        X = X[:, None].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        HM, WH, M = HM.cuda(), WH.cuda(), M.cuda()
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ph, pw = model(X)
        ph, pw = ph.float(), pw.float()
        lh = focal(ph, HM)
        lw = (F.l1_loss(pw, WH, reduction="none").sum(1) * M).sum() / M.sum().clamp(min=1)
        loss = lh + 0.1 * lw
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 200 == 0:
            print(f"step {it + 1}/{steps} loss {lh.item():.3f} wh {lw.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % 250 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": it + 1}, ckpt)
    torch.save({"model": model.state_dict(), "classes": classes}, a.out)
    print("saved", a.out)
    if va:
        evaluate(model, classes, labels, va, a)


def evaluate(model, classes, labels, va, a):
    model.eval()
    crates = pd.read_csv(a.data / "crates.csv")
    ids = [classes.index(g) for g in crates.glyph_id]
    rows = []
    for n in va:
        img = load_gray(a.data / "train" / "images" / n)
        det, _ = detect(model, img, ids, thr=0.0)
        gt = labels[labels.image == n]
        for k, g in enumerate(crates.glyph_id):
            scores = [d[1] for d in det if d[0] == k]
            rows.append((n, g, (gt.glyph_id == g).sum(), scores))
    for thr in (0.2, 0.3, 0.4, 0.5):
        tot_err, tot_gt, tot_pred = 0, 0, 0
        per = {}
        for n, g, t, sc in rows:
            p = sum(s >= thr for s in sc)
            per.setdefault(g, [0, 0, 0]); per[g][0] += t; per[g][1] += p; per[g][2] += abs(p - t)
            tot_err += abs(p - t); tot_gt += t; tot_pred += p
        print(f"thr {thr}: gt {tot_gt} pred {tot_pred} abs-err {tot_err} | " +
              " ".join(f"{g}:{v[0]}/{v[1]}/{v[2]}" for g, v in per.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", type=Path, default=Path("../task_china"))
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("detector.pt"))
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
