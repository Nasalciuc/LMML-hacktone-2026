"""Stamp segmentation on a stock-book page.

The album background is dark and constant along each row (glassine strips are
horizontal), so a per-row background colour is estimated and every pixel whose
horizontal neighbourhood differs from it (colour or texture) is "stamp".
This also catches dark stamps that a fixed colour threshold misses, and small
strip stamps (no 120 px minimum).
"""
import numpy as np
import cv2

BG = np.array([38, 36, 34], np.float32)


def stamp_masks(page, dev_thr=7.0, min_w=50, min_h=30, min_area=4000):
    f = page.astype(np.float32)
    H, W = f.shape[:2]
    d0 = np.abs(f - BG).max(2)
    m0 = cv2.morphologyEx((d0 > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m0 = cv2.dilate(m0, np.ones((15, 15), np.uint8))
    bg = np.full((H, 3), np.nan, np.float32)
    for y in range(H):
        sel = m0[y] == 0
        if sel.sum() > 50:
            bg[y] = np.median(f[y, sel], 0)
    for c in range(3):
        v = bg[:, c]; ok = ~np.isnan(v)
        bg[:, c] = np.interp(np.arange(H), np.where(ok)[0], v[ok]) if ok.any() else BG[c]
    lm = cv2.blur(f, (9, 1))
    dev = np.abs(lm - bg[:, None, :]).max(2)
    lstd = np.sqrt(np.maximum(cv2.blur(f * f, (9, 1)) - lm * lm, 0)).mean(2)
    bgstd = np.median(lstd[m0 == 0]) if (m0 == 0).any() else 2.0
    cand = (dev > dev_thr) | (lstd > 3.0 * bgstd + 2)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if w > min_w and h > min_h and a > min_area:
            mk = (lab == k).astype(np.uint8)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            full = np.zeros_like(mk)
            cv2.drawContours(full, cnts, -1, 1, -1)
            out.append(((int(x), int(y), int(w), int(h)), full.astype(bool)))
    return out
