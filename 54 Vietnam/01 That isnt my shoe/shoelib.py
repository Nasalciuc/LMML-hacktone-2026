"""Shared code: network, GPU augmentation, training loop, open-set scores."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.backends.cudnn.benchmark = True
CLASSES = ["nike", "adidas", "converse"]
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- network
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
        self.feat_dim = cin

    def features(self, x):
        x = self.layers(self.stem(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x, return_feat=False):
        f = self.features(x)
        logits = self.fc(self.drop(f))
        return (logits, f) if return_feat else logits


# ----------------------------------------------------------------------------- data
def load_images(folder: Path, names) -> torch.Tensor:
    arr = np.stack([np.asarray(Image.open(folder / n).convert("RGB").resize((256, 256), Image.BILINEAR)) for n in names])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,256,256


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def prep_eval(x_u8: torch.Tensor, res: int) -> torch.Tensor:
    x = x_u8.float() / 255.0
    x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    return normalize(x)


# ----------------------------------------------------------------------------- GPU augmentation
def augment(x_u8: torch.Tensor, res: int, strong: bool = True) -> torch.Tensor:
    """Random resized crop + flip + rotation via one affine warp, colour jitter, grayscale, erasing."""
    n = x_u8.shape[0]
    dev = x_u8.device
    x = x_u8.float() / 255.0
    scale = torch.empty(n, device=dev).uniform_(0.35, 1.0)        # area fraction
    logr = torch.empty(n, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    ar = torch.exp(logr)
    sw = torch.sqrt(scale * ar).clamp(max=1.0)
    sh = torch.sqrt(scale / ar).clamp(max=1.0)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - sw)
    ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - sh)
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(20)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = cos * sw * flip
    theta[:, 0, 1] = -sin * sh
    theta[:, 1, 0] = sin * sw * flip
    theta[:, 1, 1] = cos * sh
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    # anti-alias first when the crop is downsampled
    x = F.interpolate(x, size=(res * 2 if res * 2 < 256 else 256,) * 2, mode="bilinear", antialias=True, align_corners=False)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if strong:
        b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.35
        c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.35
        s = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.5
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * s
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * c + m
        x = x * b
        # hue rotation (small) through channel mixing in YIQ space
        h = (torch.rand(n, device=dev) * 2 - 1) * math.radians(25)
        x = hue_rotate(x, h)
        g = torch.rand(n, 1, 1, 1, device=dev) < 0.15
        x = torch.where(g, x.mean(1, keepdim=True).expand_as(x), x)
        x = x.clamp(0, 1)
        # random erasing
        er = torch.rand(n, device=dev) < 0.4
        if er.any():
            ys = torch.arange(res, device=dev).view(1, res, 1)
            xs = torch.arange(res, device=dev).view(1, 1, res)
            eh = (torch.rand(n, device=dev) * 0.4 + 0.1) * res
            ew = (torch.rand(n, device=dev) * 0.4 + 0.1) * res
            ey = torch.rand(n, device=dev) * (res - eh)
            ex = torch.rand(n, device=dev) * (res - ew)
            box = (ys >= ey.view(-1, 1, 1)) & (ys < (ey + eh).view(-1, 1, 1)) & (xs >= ex.view(-1, 1, 1)) & (xs < (ex + ew).view(-1, 1, 1))
            box = box & er.view(-1, 1, 1)
            x = torch.where(box.unsqueeze(1), torch.rand_like(x), x)
    return normalize(x)


def hue_rotate(x, h):
    T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=x.device)
    Ti = torch.linalg.inv(T)
    yiq = torch.einsum("ij,njhw->nihw", T, x)
    c, s = torch.cos(h).view(-1, 1, 1), torch.sin(h).view(-1, 1, 1)
    i2 = yiq[:, 1] * c - yiq[:, 2] * s
    q2 = yiq[:, 1] * s + yiq[:, 2] * c
    yiq = torch.stack([yiq[:, 0], i2, q2], 1)
    return torch.einsum("ij,njhw->nihw", Ti, yiq)


# ----------------------------------------------------------------------------- synthetic outliers
def make_outliers(x_norm: torch.Tensor, kind_p=(0.5, 0.5)) -> torch.Tensor:
    """Destroy the brand details but keep colours / rough shape: patch shuffle or heavy blur."""
    n, c, h, w = x_norm.shape
    out = x_norm.clone()
    k = torch.rand(n, device=x_norm.device)
    sh = k < kind_p[0]
    if sh.any():
        g = 4
        ph, pw = h // g, w // g
        p = out[sh].unfold(2, ph, ph).unfold(3, pw, pw)  # n,c,g,g,ph,pw
        m = p.shape[0]
        p = p.reshape(m, c, g * g, ph, pw)
        perm = torch.rand(m, g * g, device=x_norm.device).argsort(1)
        p = torch.gather(p, 2, perm.view(m, 1, g * g, 1, 1).expand(m, c, g * g, ph, pw))
        p = p.view(m, c, g, g, ph, pw).permute(0, 1, 2, 4, 3, 5).reshape(m, c, h, w)
        out[sh] = p
    bl = ~sh
    if bl.any():
        small = F.interpolate(out[bl], size=(h // 10, w // 10), mode="bilinear", antialias=True, align_corners=False)
        out[bl] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return out


# ----------------------------------------------------------------------------- training
def train_model(x_u8, y, n_classes, *, res=160, epochs=300, bs=64, lr=2e-3, wd=5e-4, width=64,
                oe_weight=0.0, mixup=0.0, label_smooth=0.0, seed=0, device="cuda", verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ShoeNet(n_classes, width).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(y)
    steps_per_epoch = max(1, n // bs)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    x_u8 = x_u8.to(device)
    y = y.to(device)
    model.train()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(x_u8[idx], res).contiguous(memory_format=torch.channels_last)
            yb = y[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if mixup > 0 and np.random.rand() < 0.5:
                    lam = float(np.random.beta(mixup, mixup))
                    j = torch.randperm(len(idx), device=device)
                    xm = lam * xb + (1 - lam) * xb[j]
                    logits = model(xm)
                    loss = lam * F.cross_entropy(logits.float(), yb, label_smoothing=label_smooth) + \
                        (1 - lam) * F.cross_entropy(logits.float(), yb[j], label_smoothing=label_smooth)
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits.float(), yb, label_smoothing=label_smooth)
                if oe_weight > 0:
                    xo = make_outliers(xb[: bs // 2])
                    lo = model(xo).float()
                    loss = loss + oe_weight * (-(lo.log_softmax(1)).mean(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        if verbose and (ep + 1) % 50 == 0:
            print(f"  ep {ep+1} loss {loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def infer(model, x_u8, res=160, device="cuda", bs=128, tta=True):
    """Returns logits (TTA-averaged) and penultimate features."""
    model.eval()
    outs, feats = [], []
    for i in range(0, len(x_u8), bs):
        xb = prep_eval(x_u8[i:i + bs].to(device), res).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, f = model(xb, return_feat=True)
            if tta:
                lg2, f2 = model(torch.flip(xb, dims=[3]), return_feat=True)
                lg = (lg + lg2) / 2
                f = (f + f2) / 2
        outs.append(lg.float())
        feats.append(f.float())
    return torch.cat(outs), torch.cat(feats)


def class_means(feats, y, n_classes):
    fn = F.normalize(feats, dim=1)
    return torch.stack([F.normalize(fn[y == c].mean(0), dim=0) for c in range(n_classes)])


def scores(logits_list, feats_list, means_list):
    """Ensemble open-set scores. Higher = more 'known'."""
    probs = torch.stack([l.softmax(1) for l in logits_list]).mean(0)
    msp, pred = probs.max(1)
    energy = torch.stack([torch.logsumexp(l, 1) for l in logits_list]).mean(0)
    cos = torch.stack([(F.normalize(f, dim=1) @ m.T) for f, m in zip(feats_list, means_list)]).mean(0)
    cosmax = cos.max(1).values
    return {"pred": pred, "msp": msp, "energy": energy, "cos": cosmax, "probs": probs}
