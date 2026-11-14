"""Forgery localisation network, trained from scratch.

Training data
  * synthetic edits applied to stamps cut from the album pages: digit swap (a digit-sized
    patch copied sideways), element clone (a small blob copied elsewhere), inscription removal
    (rectangle filled with its border colour). Each edit is done at a higher resolution, then the
    stamp is resampled back and re-saved as JPEG, like a stamp pasted into a scanned page.
  * the 30 exhibits with their real masks (a few held out for validation).

    python forgery_net.py train --data ../task_kazakhstan --val 8
    python forgery_net.py train --data ../task_kazakhstan --val 0      # final model
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BG = np.array([38, 36, 34])


def stamps_on(page):
    """Boxes (x, y, w, h) of the stamps: everything that is not the dark album background."""
    d = np.abs(page.astype(np.int16) - BG).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [tuple(int(v) for v in s[:4]) for s in stats[1:] if s[2] > 120 and s[3] > 120]


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], 1)


# ---------------------------------------------------------------- synthetic edits
def edit_digit(img, rng):
    H, W = img.shape[:2]
    h = int(rng.uniform(0.08, 0.2) * H); w = int(h * rng.uniform(0.45, 0.8))
    x = rng.integers(10, max(11, W - 2 * w - 10)); y = rng.integers(10, max(11, H - h - 10))
    dx = int(w * rng.uniform(0.85, 1.25)) * (1 if rng.random() < 0.5 else -1)
    sx = x + w if dx > 0 else x
    tx = int(np.clip(sx + dx, 0, W - w)); src = img[y:y + h, sx:sx + w].copy()
    if src.shape[:2] != (h, w):
        return None
    out = img.copy(); m = np.zeros((H, W), np.float32)
    if rng.random() < 0.5:
        out[y:y + h, tx:tx + w] = src; m[y:y + h, tx:tx + w] = 1
    else:  # paste only the strokes (pixels far from the patch's border colour)
        border = np.concatenate([src[0], src[-1], src[:, 0], src[:, -1]]).astype(np.float32)
        bc = np.median(border, 0)
        fg = (np.abs(src.astype(np.float32) - bc).max(2) > 40).astype(np.float32)
        fg = cv2.GaussianBlur(cv2.dilate(fg, np.ones((3, 3))), (3, 3), 0)[..., None]
        if fg.mean() < 0.05:
            return None
        out[y:y + h, tx:tx + w] = (fg * src + (1 - fg) * out[y:y + h, tx:tx + w]).astype(np.uint8)
        m[y:y + h, tx:tx + w] = 1
    return out, m


def edit_clone(img, rng):
    H, W = img.shape[:2]
    r = int(rng.uniform(0.03, 0.08) * min(H, W)) + 4
    cx, cy = rng.integers(r, W - r), rng.integers(r, H - r)
    tx, ty = rng.integers(r, W - r), rng.integers(r, H - r)
    if abs(tx - cx) < 2 * r and abs(ty - cy) < 2 * r:
        return None
    blob = np.zeros((2 * r, 2 * r), np.float32)
    cv2.ellipse(blob, (r, r), (int(r * rng.uniform(0.6, 1)), int(r * rng.uniform(0.6, 1))), rng.uniform(0, 180), 0, 360, 1, -1)
    blob = cv2.GaussianBlur(blob, (5, 5), 0)[..., None]
    out = img.copy()
    src = img[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
    dst = out[ty - r:ty + r, tx - r:tx + r].astype(np.float32)
    out[ty - r:ty + r, tx - r:tx + r] = (blob * src + (1 - blob) * dst).astype(np.uint8)
    m = np.zeros((H, W), np.float32); m[ty - r:ty + r, tx - r:tx + r] = blob[..., 0] > 0.5
    return out, m


def edit_remove(img, rng):
    """Remove an 'inscription': a rectangle on a plain background gets its border colour."""
    H, W = img.shape[:2]
    best = None
    for _ in range(30):
        h = int(rng.uniform(0.05, 0.15) * H); w = int(rng.uniform(0.1, 0.3) * W)
        x = rng.integers(5, max(6, W - w - 5)); y = rng.integers(5, max(6, H - h - 5))
        patch = img[y:y + h, x:x + w]
        border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]).astype(np.float32)
        score = border.std(0).mean() - 0.3 * patch.reshape(-1, 3).astype(np.float32).std(0).mean()
        if best is None or score < best[0]:
            best = (score, x, y, w, h, np.median(border, 0))
    _, x, y, w, h, col = best
    out = img.copy()
    u = rng.random()
    if u < 0.75:
        fill = np.zeros((h, w, 3), np.float32) + col
        if u < 0.55:  # fresh, spatially white noise, as seen in the removal exhibits
            fill += rng.normal(0, rng.uniform(1.0, 4.0), (h, w, 1 if rng.random() < 0.5 else 3))
        out[y:y + h, x:x + w] = np.clip(fill, 0, 255).astype(np.uint8)
    else:
        mm = np.zeros((H, W), np.uint8); mm[y:y + h, x:x + w] = 1
        out = cv2.inpaint(img, mm, 3, cv2.INPAINT_TELEA)
    m = np.zeros((H, W), np.float32); m[y:y + h, x:x + w] = 1
    return out, m


EDITS = [edit_digit, edit_clone, edit_remove]


def synth(stamp, rng):
    """Edit the stamp at page resolution (most of the time) and save as JPEG, like a page."""
    H, W = stamp.shape[:2]
    resample = rng.random() < 0.25
    up = rng.uniform(1.0, 1.6) if resample else 1.0
    big = cv2.resize(stamp, (int(W * up), int(H * up)), interpolation=cv2.INTER_CUBIC) if resample else stamp
    for _ in range(10):
        r = EDITS[rng.choice(3, p=[0.4, 0.2, 0.4])](big, rng)
        if r is not None:
            break
    else:
        r = (big, np.zeros(big.shape[:2], np.float32))
    out, m = r
    if resample:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    out = jpeg(out, rng.integers(90, 99))
    return out, (m > 0.5).astype(np.float32)


# ---------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=6, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(cin, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.b = block(8 * c, 8 * c)
        self.d4, self.d3, self.d2, self.d1 = block(16 * c, 4 * c), block(8 * c, 2 * c), block(4 * c, c), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2)); b = self.b(F.max_pool2d(e4, 2))
        up = lambda t, s: F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat([up(b, e4), e4], 1)); d3 = self.d3(torch.cat([up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1)); d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.out(d1)


def features(x):
    """x: (N,3,H,W) float 0..255 on GPU -> RGB + high-pass residual (3 ch)."""
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
    res = (x - blur) / 3.0
    return torch.cat([(x - 128) / 64, res], 1)


@torch.no_grad()
def predict(model, img):
    H, W = img.shape[:2]
    Hp, Wp = (H + 15) // 16 * 16, (W + 15) // 16 * 16
    t = torch.from_numpy(img).float().permute(2, 0, 1)[None].cuda()
    t = F.pad(t, (0, Wp - W, 0, Hp - H), mode="replicate")
    p = 0
    for fl in (False, True):
        tt = torch.flip(t, (3,)) if fl else t
        o = model(features(tt)).sigmoid()
        p = p + (torch.flip(o, (3,)) if fl else o)
    return (p / 2)[0, 0, :H, :W].cpu().numpy()


def load_stamps(data):
    out = []
    for f in sorted((data / "albums").glob("*/*.jpg")):
        page = cv2.imread(str(f))
        for x, y, w, h in stamps_on(page):
            out.append(page[y:y + h, x:x + w].copy())
    return out


def load_exhibits(data):
    ex = []
    for f in sorted((data / "exhibits").glob("exhibit_??.jpg")):
        m = cv2.imread(str(f).replace(".jpg", "_mask.png"), 0)
        ex.append((f.stem, cv2.imread(str(f)), (m > 127).astype(np.float32)))
    return ex


def random_crop(img, m, rng, size=256):
    H, W = img.shape[:2]
    if H < size or W < size:
        ph, pw = max(0, size - H), max(0, size - W)
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=BG.tolist())
        m = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        H, W = img.shape[:2]
    if m.sum() > 0 and rng.random() < 0.8:  # crop containing the edit
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean(), xs.mean()
        y0 = int(np.clip(cy - rng.uniform(0.15, 0.85) * size, 0, H - size))
        x0 = int(np.clip(cx - rng.uniform(0.15, 0.85) * size, 0, W - size))
    else:
        y0, x0 = rng.integers(0, H - size + 1), rng.integers(0, W - size + 1)
    return img[y0:y0 + size, x0:x0 + size], m[y0:y0 + size, x0:x0 + size]


def train(a):
    rng = np.random.default_rng(a.seed)
    stamps = load_stamps(a.data)
    ex = load_exhibits(a.data)
    va = ex[:a.val]; tr_ex = ex[a.val:]
    print(len(stamps), "album stamps,", len(tr_ex), "training exhibits,", len(va), "validation exhibits", flush=True)
    model = UNet().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.iters, pct_start=0.05)
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(10)

    def one(seed):
        r = np.random.default_rng(seed)
        if tr_ex and r.random() < a.ex_frac:
            _, img, m = tr_ex[r.integers(len(tr_ex))]
            img = jpeg(img, r.integers(88, 99))
        else:
            img, m = synth(stamps[r.integers(len(stamps))], r)
        s = r.uniform(0.75, 1.25)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img, m = random_crop(img, m, r)
        if r.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        return np.ascontiguousarray(img), np.ascontiguousarray(m)

    def batch(seed):
        r = np.random.default_rng(seed)
        res = list(pool.map(one, r.integers(1 << 30, size=a.bs)))
        return (torch.from_numpy(np.stack([x[0] for x in res])).permute(0, 3, 1, 2).float(),
                torch.from_numpy(np.stack([x[1] for x in res]))[:, None])

    bpool = cf.ThreadPoolExecutor(2)
    futs = [bpool.submit(batch, rng.integers(1 << 30)) for _ in range(2)]
    t0 = time.time()
    start = 0
    ckpt = a.out.with_suffix(".ckpt")
    if ckpt.exists():  # resume after an interruption
        ck = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"]
        print("resumed at step", start, flush=True)
    for it in range(start, a.iters):
        X, M = futs.pop(0).result(); futs.append(bpool.submit(batch, rng.integers(1 << 30)))
        X, M = X.cuda(), M.cuda()
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(features(X).contiguous(memory_format=torch.channels_last))
        p = p.float()
        loss = F.binary_cross_entropy_with_logits(p, M, pos_weight=torch.tensor(5.0, device="cuda"))
        pr = p.sigmoid()
        dice = 1 - (2 * (pr * M).sum() + 1) / (pr.sum() + M.sum() + 1)
        loss = loss + dice
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 500 == 0:
            msg = f"iter {it + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s"
            if va:
                msg += " | val " + validate(model, va)
            print(msg, flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": it + 1}, ckpt)
    torch.save(model.state_dict(), a.out)
    print("saved", a.out)


def validate(model, va):
    model.eval()
    hits = []
    for name, img, m in va:
        p = predict(model, img)
        ps = cv2.GaussianBlur(p, (0, 0), 6)
        y, x = np.unravel_index(ps.argmax(), ps.shape)
        x0, y0, w, h = cv2.boundingRect(m.astype(np.uint8))
        hits.append(int(x0 - 10 <= x <= x0 + w + 10 and y0 - 10 <= y <= y0 + h + 10))
    return f"{sum(hits)}/{len(hits)} " + "".join(map(str, hits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", type=Path, default=Path("../task_kazakhstan"))
    ap.add_argument("--val", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--ex_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("forgery_net.pt"))
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
