"""Classical forensic cues on one stamp crop.

copy_move(): every 12x12 block of the high-pass residual is compared with every other block of
  the same stamp (cosine similarity, GPU). A cloned area shows up as a group of neighbouring
  blocks whose best match sits at the same displacement with high similarity.
flatness(): local residual energy compared with the stamp's typical grain; a removed
  inscription is filled with a flat colour and has almost no grain left.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def residual(img, edge_free=True):
    """High-pass residual; with edge_free the strong edges are zeroed, so that only grain is
    compared (two prints of the same glyph share edges, a cloned patch shares grain)."""
    g = img.astype(np.float32).mean(2)
    r = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    if edge_free:
        sm = cv2.GaussianBlur(g, (0, 0), 1.5)
        grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
        edge = cv2.dilate((grad > 40).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        r[edge] = 0
    return r


def copy_move(img, B=12, ST=2, MIND=14, margin=14):
    """Returns per-pixel map of 'clone evidence' and the per-block displacement field."""
    H, W = img.shape[:2]
    res = residual(img)
    t = torch.from_numpy(res).cuda()[None, None]
    P = F.unfold(t, B, stride=ST)[0].T
    ny, nx = (H - B) // ST + 1, (W - B) // ST + 1
    ys = (torch.arange(ny, device="cuda").repeat_interleave(nx) * ST)
    xs = (torch.arange(nx, device="cuda").repeat(ny) * ST)
    P = P - P.mean(1, keepdim=True)
    nrm = P.norm(dim=1)
    Pn = (P / (nrm[:, None] + 1e-6)).half()
    best = torch.full((len(P),), -1.0, device="cuda")
    dy = torch.zeros(len(P), device="cuda"); dx = torch.zeros(len(P), device="cuda")
    for i in range(0, len(P), 2048):
        s = Pn[i:i + 2048] @ Pn.T
        v, j = s.topk(16, dim=1)
        ddy = ys[j] - ys[i:i + 2048, None]; ddx = xs[j] - xs[i:i + 2048, None]
        far = (ddy.abs() >= MIND) | (ddx.abs() >= MIND)
        v = torch.where(far, v.float(), torch.full_like(v.float(), -1))
        k = v.argmax(1)
        best[i:i + 2048] = v.gather(1, k[:, None])[:, 0]
        dy[i:i + 2048] = ddy.gather(1, k[:, None])[:, 0].float()
        dx[i:i + 2048] = ddx.gather(1, k[:, None])[:, 0].float()
    best = best.view(ny, nx).cpu().numpy(); dy = dy.view(ny, nx).cpu().numpy(); dx = dx.view(ny, nx).cpu().numpy()
    energy = nrm.view(ny, nx).cpu().numpy() / B
    # coherence: fraction of 5x5 neighbours with the same displacement (+-2 px)
    coh = np.zeros_like(best)
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            sdy = np.roll(np.roll(dy, oy, 0), ox, 1); sdx = np.roll(np.roll(dx, oy, 0), ox, 1)
            coh += (np.abs(sdy - dy) <= 2) & (np.abs(sdx - dx) <= 2)
    coh /= 25
    ev = np.clip(best, 0, 1) * coh * (energy > 0.4)
    m = int(np.ceil(margin / ST))
    ev[:m] = ev[-m:] = 0; ev[:, :m] = ev[:, -m:] = 0
    full = cv2.resize(ev, (nx * ST, ny * ST), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W), np.float32)
    out[B // 2:B // 2 + full.shape[0], B // 2:B // 2 + full.shape[1]] = full[:H - B // 2, :W - B // 2]
    return out, (best, dy, dx, energy, coh)


def flatness(img, margin=14, win=21):
    """High where the residual is much smaller than the stamp's usual grain (inside the stamp)."""
    res = residual(img)
    e = np.sqrt(cv2.blur(res ** 2, (win, win)))
    grain = np.median(e[margin:-margin, margin:-margin])
    out = np.clip(1 - e / (grain + 1e-6), 0, 1)
    out[:margin] = out[-margin:] = 0; out[:, :margin] = out[:, -margin:] = 0
    return out, grain
