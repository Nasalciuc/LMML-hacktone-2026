"""Pretrained backbone + 4th class `unknown`: model, GPU augmentation, training loop."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CLASSES = ["adidas", "converse", "nike", "unknown"]
KNOWN = CLASSES[:3]
RES = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# brands that are really one of the known ones (or a sub-label) -> never used as `unknown`
NOT_UNKNOWN = {"nike", "adidas", "converse", "jordan", "off-white"}  # off-white = mostly Nike collabs


def build(arch="efficientnet_b0", drop=0.3, n_out=4):
    if arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="DEFAULT")
        m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(1280, n_out))
    elif arch == "mobilenet_v3_large":
        m = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
        m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(960, n_out))
    elif arch == "regnet_y_400mf":
        m = torchvision.models.regnet_y_400mf(weights="DEFAULT")
        m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(440, n_out))
    else:
        raise ValueError(arch)
    return m


def to_input(x_u8):
    """uint8 N,3,256,256 (on device) -> normalised N,3,224,224. Same math as the exported forward()."""
    x = x_u8.float() / 255.0
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def _blur(x, sigma):
    k = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)


def augment(x):
    """x: float 0-1, N,3,224,224 on GPU. Random resized crop / flip / small rotation, colour jitter, erasing."""
    n, dev = x.shape[0], x.device
    s = torch.empty(n, device=dev).uniform_(0.6, 1.0)          # zoom (side fraction kept)
    ar = torch.exp(torch.empty(n, device=dev).uniform_(-0.2, 0.2))
    sx, sy = s * ar.sqrt(), s / ar.sqrt()
    tx = (1 - sx).clamp(min=0) * torch.empty(n, device=dev).uniform_(-1, 1)
    ty = (1 - sy).clamp(min=0) * torch.empty(n, device=dev).uniform_(-1, 1)
    ang = torch.empty(n, device=dev).uniform_(-15, 15) * math.pi / 180
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    c, si = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c * sx * flip, -si * sy, tx], 1),
                         torch.stack([si * sx * flip, c * sy, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # colour
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    ct = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    sat = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    g = x.mean(1, keepdim=True)
    x = g + (x - g) * sat
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m) * b
    gray = torch.rand(n, 1, 1, 1, device=dev) < 0.1
    x = torch.where(gray, x.mean(1, keepdim=True).expand_as(x), x)
    x = x.clamp(0, 1)
    # low-level statistics (sharpness, noise) shaken for every image alike, so that the source of a
    # photo (task set vs. web scrape) is not something the net can use
    r = torch.rand(n, 1, 1, 1, device=dev)
    blur = _blur(x, float(torch.empty(1).uniform_(0.5, 1.2)))
    amt = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.3, 1.0)
    x = torch.where(r < 0.25, blur, torch.where(r < 0.45, (x + amt * (x - blur)), x))
    noise = torch.rand(n, 1, 1, 1, device=dev) < 0.25
    x = torch.where(noise, x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.005, 0.02), x)
    x = x.clamp(0, 1)
    # random erasing (small, 25 % of images) - keeps the net from relying on one logo spot
    er = torch.rand(n, device=dev) < 0.25
    if er.any():
        hh = (torch.rand(n, device=dev) * 0.25 + 0.1) * RES
        ww = (torch.rand(n, device=dev) * 0.25 + 0.1) * RES
        cy, cx = torch.rand(n, device=dev) * RES, torch.rand(n, device=dev) * RES
        yy = torch.arange(RES, device=dev).view(1, RES, 1)
        xx = torch.arange(RES, device=dev).view(1, 1, RES)
        box = ((yy - cy.view(-1, 1, 1)).abs() < hh.view(-1, 1, 1) / 2) & ((xx - cx.view(-1, 1, 1)).abs() < ww.view(-1, 1, 1) / 2)
        box = box & er.view(-1, 1, 1)
        x = torch.where(box.unsqueeze(1), torch.rand(n, 3, 1, 1, device=dev), x)
    return x


def train_model(xk, yk, xu, arch="efficientnet_b0", epochs=15, bs=64, lr=4e-4, wd=0.05, u_ratio=1.0,
                smooth=0.1, seed=0, dev="cuda", log=None, n_out=4, loss_kind="ce"):
    """xk: uint8 N,3,256,256 knowns (cpu), yk: int labels 0..2, xu: uint8 outliers -> label 3.
    Each epoch = all knowns + u_ratio * len(knowns) random outliers."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch, n_out=n_out).to(dev).to(memory_format=torch.channels_last)
    head = [p for n, p in m.named_parameters() if n.startswith(("classifier", "fc"))]
    body = [p for n, p in m.named_parameters() if not n.startswith(("classifier", "fc"))]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * 5}], weight_decay=wd)
    nk = len(xk); nu = int(nk * u_ratio) if len(xu) else 0
    steps = epochs * math.ceil((nk + nu) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 5], total_steps=steps, pct_start=0.15)
    yk = torch.as_tensor(yk)
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        m.train()
        ui = rng.choice(len(xu), nu, replace=nu > len(xu)) if nu else np.zeros(0, int)
        X_idx = np.concatenate([np.arange(nk), nk + np.arange(nu)])
        rng.shuffle(X_idx)
        tot, correct, cnt = 0.0, 0, 0
        for i in range(0, len(X_idx), bs):
            b = X_idx[i:i + bs]
            kb, ub = b[b < nk], ui[b[b >= nk] - nk]
            xb = torch.cat([xk[kb], xu[ub]]).to(dev, non_blocking=True)
            yb = torch.cat([yk[kb], torch.full((len(ub),), 3, dtype=torch.long)]).to(dev)
            x = xb.float() / 255.0
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
            x = augment(x)
            x = ((x - MEAN.to(dev)) / STD.to(dev)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
            if loss_kind == "bce":      # one sigmoid per brand: an unknown shoe can score low on all of them
                loss = F.binary_cross_entropy_with_logits(out.float(), F.one_hot(yb, out.shape[1]).float())
            else:
                loss = F.cross_entropy(out.float(), yb, label_smoothing=smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            tot += loss.item() * len(b); cnt += len(b); correct += (out.argmax(1) == yb).sum().item()
        if log:
            log(f"  ep {ep+1}/{epochs} loss {tot/cnt:.4f} train acc {correct/cnt:.3f}")
    m.eval()
    return m


@torch.no_grad()
def predict_logits(m, x_u8, dev="cuda", bs=128, tta=True):
    outs = []
    for i in range(0, len(x_u8), bs):
        x = to_input(x_u8[i:i + bs].to(dev)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(x).float()
            if tta:
                o = (o + m(x.flip(3)).float()) / 2
        outs.append(o.cpu())
    return torch.cat(outs)


@torch.no_grad()
def predict_feats(m, x_u8, dev="cuda", bs=128):
    """Penultimate features (input of the last linear layer), flip-averaged."""
    head = m.classifier if hasattr(m, "classifier") else m.fc
    store = []
    h = head.register_forward_pre_hook(lambda mod, inp: store.append(inp[0].float()))
    outs = []
    for i in range(0, len(x_u8), bs):
        x = to_input(x_u8[i:i + bs].to(dev)).contiguous(memory_format=torch.channels_last)
        store.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x); m(x.flip(3))
        outs.append(((store[0] + store[1]) / 2).cpu())
    h.remove()
    return torch.cat(outs)


def decide(logits, tau=0.0, bias=0.0):
    """Final rule (same as exported model): add `bias` to the unknown logit, softmax over 4;
    unknown if argmax is `unknown` or the best known-class probability is below tau."""
    logits = torch.cat([logits[:, :3], logits[:, 3:] + bias], 1)
    p = logits.softmax(1)
    pk, ik = p[:, :3].max(1)
    return torch.where((p[:, 3] >= pk) | (pk < tau), torch.full_like(ik, 3), ik)
