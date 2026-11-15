"""PCE of a batch of cross-correlation maps (shared by pce_pairs.py, pce_scales.py)."""
import torch


def pce_batch(c):
    B, H, W = c.shape
    flat = c.reshape(B, -1); idx = flat.abs().argmax(1); pk = flat.gather(1, idx[:, None])[:, 0]
    py, px = idx // W, idx % W
    tot = (flat * flat).sum(1)
    oy = torch.arange(-5, 6, device=c.device); yy = (py[:, None] + oy) % H; xx = (px[:, None] + oy) % W
    win = c[torch.arange(B, device=c.device)[:, None, None], yy[:, :, None], xx[:, None, :]]
    e = (tot - (win * win).sum((1, 2))) / (H * W - 121)
    dy = torch.where(py > H // 2, py - H, py); dx = torch.where(px > W // 2, px - W, px)
    return (pk * pk / e), pk, dy, dx
