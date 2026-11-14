"""Segment the shell in every image and resample it into a canonical frame.

Output (npz): names, crops (N,H,W,3 uint8, shell upright, background zeroed),
masks (N,H,W uint8), geom (N, k) simple geometry numbers.
"""
import os
import sys

import cv2
import numpy as np

H, W = 256, 80


def shell_mask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h >= 30) & (h <= 90) & (s >= 60)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = [1 + k for k in range(len(areas)) if areas[k] >= 0.03 * big]
    m = np.isin(lab, keep).astype(np.uint8) * 255
    # close the grey band and small gaps, then fill holes
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 61)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def canonical(im, m):
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    ev, evec = np.linalg.eigh(cov)
    ax = evec[:, 1]                         # main axis
    if ax[1] < 0:
        ax = -ax
    ang = np.degrees(np.arctan2(ax[0], ax[1]))  # tilt from vertical
    # project to get extent along the axis / across it
    perp = np.array([ax[1], -ax[0]])
    t = (pts - c) @ ax
    u = (pts - c) @ perp
    t0, t1 = t.min(), t.max()
    u0, u1 = u.min(), u.max()
    L = t1 - t0
    # robust width: median over cross-sections of the cylindrical part
    tb = np.round(t - t0).astype(int)
    sel = (tb > 0.45 * L) & (tb < 0.75 * L)
    if sel.sum() > 50:
        cnt = np.bincount(tb[sel])
        Wd = float(np.median(cnt[cnt > 0]))
        uc = float(np.median(u[sel]))
    else:
        Wd = u1 - u0
        uc = (u0 + u1) / 2
    c = c + perp * uc
    # nose direction: the narrower end is the tip
    q = L * 0.15
    w_lo = np.ptp(u[t < t0 + q]) if (t < t0 + q).any() else 0
    w_hi = np.ptp(u[t > t1 - q]) if (t > t1 - q).any() else 0
    tip_top = w_lo < w_hi                    # small t = upper in image
    # affine: output (x,y) -> input point
    # y_out in [0,H) maps t from tip to base; x_out maps u
    half = max(Wd / 2, 1) * 1.25
    sy = L * 1.04 / H
    sx = 2 * half / W
    dir_t = ax if tip_top else -ax
    start = c + dir_t * ((t0 if tip_top else -t1) - L * 0.02)
    A = np.zeros((2, 3))
    A[:, 1] = dir_t * sy
    A[:, 0] = perp * sx * (1 if tip_top else -1)
    A[:, 2] = start - A[:, 0] * (W / 2)
    crop = cv2.warpAffine(im, A, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mc = cv2.warpAffine(m, A, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
    geom = [L, Wd, ang, c[0], c[1], float(len(xs)), float(tip_top)]
    return crop, mc, geom


def main(src, out):
    names = sorted(f for f in os.listdir(src) if f.lower().endswith(".png") and "(" not in f)
    crops, masks, geoms = [], [], []
    for f in names:
        im = cv2.imread(os.path.join(src, f), cv2.IMREAD_COLOR)
        m = shell_mask(im)
        cr, mc, g = canonical(im, m)
        crops.append(cr)
        masks.append(mc)
        geoms.append(g)
    np.savez_compressed(out, names=np.array(names), crops=np.array(crops), masks=np.array(masks), geom=np.array(geoms))
    print(len(names), "done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
