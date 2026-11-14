"""Noise-level inconsistency. Every stamp carries its own scanner grain. A painted-over
inscription has no grain (or fresh synthetic noise), a pasted digit is smoother than the
print around it. Grain is measured only on flat pixels (edges masked), compared with what
the same stamp shows at the same brightness, and the log-ratio is the anomaly."""
import cv2
import numpy as np


def noise_anomaly(stamp, win=15, margin=12):
    f = stamp.astype(np.float32)
    g = f.mean(2)
    res = np.zeros_like(g)
    for c in range(3):
        ch = f[..., c]
        res += np.abs(ch - cv2.blur(ch, (3, 3)))
    res /= 3
    sm = cv2.GaussianBlur(g, (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
    flat = (grad < np.percentile(grad, 60)).astype(np.float32)
    H, W = g.shape
    flat[:margin] = flat[-margin:] = 0; flat[:, :margin] = flat[:, -margin:] = 0
    den = cv2.blur(flat, (win, win))
    lvl = cv2.blur(res * flat, (win, win)) / (den + 1e-6)
    bright = cv2.blur(g * flat, (win, win)) / (den + 1e-6)
    ok = den > 0.3
    # expected grain as a function of brightness, per stamp
    bins = np.clip((bright / 32).astype(int), 0, 7)
    exp = np.zeros(8, np.float32)
    for b in range(8):
        sel = ok & (bins == b)
        exp[b] = np.median(lvl[sel]) if sel.sum() > 200 else np.median(lvl[ok])
    expected = exp[bins]
    a = np.abs(np.log((lvl + 0.05) / (expected + 0.05)))
    a[~ok] = 0
    return a
