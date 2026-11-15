"""Per-stamp noise-level-function residuals.

Inside one stamp, the expected noise statistics depend on the local content
(colour, brightness, edges).  We learn that dependence from the stamp itself
(the altered region is ~1-2 % of its area) and keep what the content does NOT
explain: too smooth (inpainting / flat fill) or too noisy / decorrelated
(synthetic noise added over a pasted or filled patch).
"""
import numpy as np
import cv2
from sklearn.ensemble import HistGradientBoostingRegressor

KCB = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 4.0
EPS = 1e-2


def box(a, w):
    return cv2.blur(a, (w, w), borderType=cv2.BORDER_REFLECT)


def cond_and_targets(img):
    f = img.astype(np.float32)
    Y = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
    C = {}
    for c in range(3):
        C[f'm{c}'] = box(f[:, :, c], 5)
    C['ystd5'] = np.sqrt(np.maximum(box(Y * Y, 5) - box(Y, 5) ** 2, 0))
    gs = cv2.GaussianBlur(Y, (0, 0), 1.5)
    gm = np.sqrt(cv2.Sobel(gs, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gs, cv2.CV_32F, 0, 1) ** 2)
    C['grad'] = gm
    C['grad9'] = box(gm, 9)
    C['lap'] = np.abs(cv2.Laplacian(cv2.GaussianBlur(Y, (0, 0), 2.5), cv2.CV_32F))
    C['satH'] = box((f.max(2) > 249).astype(np.float32), 5)
    C['satL'] = box((f.min(2) < 6).astype(np.float32), 5)
    r = np.dstack([cv2.filter2D(f[:, :, c], -1, KCB, borderType=cv2.BORDER_REFLECT) for c in range(3)])
    ry = cv2.filter2D(Y, -1, KCB, borderType=cv2.BORDER_REFLECT)
    T = {}
    T['lvar'] = np.log(box(ry * ry, 5) + EPS)
    # decorrelated (chroma-like) residual energy: residual minus its channel mean
    rc = r - r.mean(2, keepdims=True)
    T['lchr'] = np.log(box((rc ** 2).sum(2), 5) + EPS)
    p = {}
    for i in range(3):
        for j in range(i, 3):
            p[(i, j)] = box(r[:, :, i] * r[:, :, j], 7)
    sd = [np.sqrt(p[(i, i)] + EPS) for i in range(3)]
    T['corr'] = (p[(0, 1)] / (sd[0] * sd[1]) + p[(0, 2)] / (sd[0] * sd[2]) + p[(1, 2)] / (sd[1] * sd[2])) / 3
    T['flat'] = box((np.abs(r).max(2) < 0.75).astype(np.float32), 5)
    rY = Y - cv2.GaussianBlur(Y, (0, 0), 1.5)
    v = box(rY * rY, 7) + EPS
    rx = np.zeros_like(rY); rx[:, :-1] = rY[:, :-1] * rY[:, 1:]
    ryy = np.zeros_like(rY); ryy[:-1, :] = rY[:-1, :] * rY[1:, :]
    T['ac'] = (box(rx, 7) + box(ryy, 7)) / (2 * v)
    # medium-scale grain: band-pass energy (sigma 1..2.5) relative to fine
    b1 = cv2.GaussianBlur(Y, (0, 0), 1.0) - cv2.GaussianBlur(Y, (0, 0), 2.5)
    T['lband'] = np.log(box(b1 * b1, 5) + EPS)
    return C, T


TKEYS = ['lvar', 'lchr', 'corr', 'flat', 'ac', 'lband']
CKEYS = ['m0', 'm1', 'm2', 'ystd5', 'grad', 'grad9', 'lap', 'satH', 'satL']


def _bin_codes(C, idx):
    y = C['m0'][idx[:, 0], idx[:, 1]] * 0.114 + C['m1'][idx[:, 0], idx[:, 1]] * 0.587 + C['m2'][idx[:, 0], idx[:, 1]] * 0.299
    yb = np.clip((y / 256 * 16).astype(np.int32), 0, 15)
    g = C['grad9'][idx[:, 0], idx[:, 1]]
    gb = np.digitize(g, [2.0, 6.0, 15.0, 35.0])
    ch = np.abs(C['m2'][idx[:, 0], idx[:, 1]] - C['m1'][idx[:, 0], idx[:, 1]]) + np.abs(C['m0'][idx[:, 0], idx[:, 1]] - C['m1'][idx[:, 0], idx[:, 1]])
    cb = np.digitize(ch, [15.0, 50.0])
    sat = (C['satH'][idx[:, 0], idx[:, 1]] > 0.5).astype(np.int32)
    fine = ((yb * 5 + gb) * 3 + cb) * 2 + sat
    coarse = yb * 5 + gb
    return fine, coarse


def nlf_residuals(img, stamp_list, seed=0):
    """binned conditional-median noise-level function per stamp; returns residual maps"""
    import pandas as pd
    C, T = cond_and_targets(img)
    H, W = img.shape[:2]
    R = {k: np.zeros((H, W), np.float32) for k in TKEYS}
    for _, mk in stamp_list:
        inner = cv2.erode(mk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        idx = np.argwhere(inner)
        if len(idx) < 500:
            continue
        fine, coarse = _bin_codes(C, idx)
        df = pd.DataFrame({'f': fine, 'c': coarse})
        cnt = df.groupby('f')['f'].transform('size').values
        for t in TKEYS:
            v = T[t][idx[:, 0], idx[:, 1]]
            df['v'] = v
            mf = df.groupby('f')['v'].transform('median').values
            mc = df.groupby('c')['v'].transform('median').values
            pred = np.where(cnt >= 60, mf, mc)
            R[t][idx[:, 0], idx[:, 1]] = v - pred
    return R, C, T


def nlf_features(img, stamp_list):
    R, C, T = nlf_residuals(img, stamp_list)
    F = {}
    for t in TKEYS:
        for w in (9, 21, 41):
            F[f'nr_{t}_{w}'] = box(R[t], w)
        F[f'nr_{t}_abs21'] = box(np.abs(R[t]), 21)
        F[f'nr_{t}_cs'] = box(R[t], 21) - box(R[t], 81)
    return F
