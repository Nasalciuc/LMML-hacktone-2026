"""Per-pixel forensic features for stamp tamper localisation.

All features are local statistics of noise residuals (how the image noise looks),
plus a few content descriptors so a classifier can tell natural smooth/textured
areas from tampered ones, plus context (centre minus surround, centre minus stamp).
"""
import numpy as np
import cv2

BG = np.array([38, 36, 34])
KCB = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 4.0
EPS = 1e-3


def stamp_masks(page, min_size=120):
    d = np.abs(page.astype(np.int16) - BG).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if w > min_size and h > min_size:
            mk = (lab == k).astype(np.uint8)
            # fill holes (stamps with dark areas inside)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            full = np.zeros_like(mk)
            cv2.drawContours(full, cnts, -1, 1, -1)
            out.append(((x, y, w, h), full.astype(bool)))
    return out


def box(a, w):
    return cv2.blur(a, (w, w), borderType=cv2.BORDER_REFLECT)


def eig3(c00, c01, c02, c11, c12, c22):
    H, W = c00.shape
    C = np.empty((H * W, 3, 3), np.float32)
    C[:, 0, 0] = c00.ravel(); C[:, 1, 1] = c11.ravel(); C[:, 2, 2] = c22.ravel()
    C[:, 0, 1] = C[:, 1, 0] = c01.ravel()
    C[:, 0, 2] = C[:, 2, 0] = c02.ravel()
    C[:, 1, 2] = C[:, 2, 1] = c12.ravel()
    ev = np.linalg.eigvalsh(C)
    return np.maximum(ev, 0).reshape(H, W, 3)


def noise_feats(r, w):
    """local 3x3 covariance of residual r (H,W,3) in a w-window -> eigen features + correlation"""
    p = {}
    for i in range(3):
        for j in range(i, 3):
            p[(i, j)] = box(r[:, :, i] * r[:, :, j], w)
    ev = eig3(p[(0, 0)], p[(0, 1)], p[(0, 2)], p[(1, 1)], p[(1, 2)], p[(2, 2)])
    major = np.log(ev[:, :, 2] + EPS)
    minor = np.log(ev[:, :, 0] + ev[:, :, 1] + EPS)
    sd = [np.sqrt(p[(i, i)] + EPS) for i in range(3)]
    corr = (p[(0, 1)] / (sd[0] * sd[1]) + p[(0, 2)] / (sd[0] * sd[2]) + p[(1, 2)] / (sd[1] * sd[2])) / 3
    return major, minor, minor - major, corr


def compute(page):
    """returns dict name -> float32 map (H,W)"""
    f = page.astype(np.float32)
    F = {}
    Y = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
    ycc = cv2.cvtColor(page, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    rcb = np.dstack([cv2.filter2D(f[:, :, c], -1, KCB, borderType=cv2.BORDER_REFLECT) for c in range(3)])
    rg = f - cv2.GaussianBlur(f, (0, 0), 1.0)
    for w in (7, 15, 31):
        a, b, c, d = noise_feats(rcb, w)
        F[f'cb_major_{w}'] = a; F[f'cb_minor_{w}'] = b; F[f'cb_ratio_{w}'] = c; F[f'cb_corr_{w}'] = d
    for w in (15,):
        a, b, c, d = noise_feats(rg, w)
        F[f'g_major_{w}'] = a; F[f'g_minor_{w}'] = b; F[f'g_ratio_{w}'] = c; F[f'g_corr_{w}'] = d
    # chroma high-frequency energy (YCrCb)
    for ci, nm in ((1, 'cr'), (2, 'cb')):
        rr = cv2.filter2D(ycc[:, :, ci], -1, KCB, borderType=cv2.BORDER_REFLECT)
        F[f'hf_{nm}_15'] = np.log(box(rr * rr, 15) + EPS)
    ry = cv2.filter2D(ycc[:, :, 0], -1, KCB, borderType=cv2.BORDER_REFLECT)
    F['hf_y_15'] = np.log(box(ry * ry, 15) + EPS)
    # spatial autocorrelation of luma residual (white noise ~ 0)
    rY = Y - cv2.GaussianBlur(Y, (0, 0), 1.5)
    v = box(rY * rY, 15) + EPS
    rx = np.zeros_like(rY); rx[:, :-1] = rY[:, :-1] * rY[:, 1:]
    ryy = np.zeros_like(rY); ryy[:-1, :] = rY[:-1, :] * rY[1:, :]
    F['ac_x_15'] = box(rx, 15) / v
    F['ac_y_15'] = box(ryy, 15) / v
    # content
    F['Y_15'] = box(Y, 15)
    F['Ystd_15'] = np.sqrt(np.maximum(box(Y * Y, 15) - F['Y_15'] ** 2, 0))
    F['chroma_15'] = box(np.abs(ycc[:, :, 1] - 128) + np.abs(ycc[:, :, 2] - 128), 15)
    F['sat_15'] = box(((Y > 251) | (Y < 4)).astype(np.float32), 15)
    gs = cv2.GaussianBlur(Y, (0, 0), 2.0)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1)
    F['grad_15'] = np.log(box(np.sqrt(gx * gx + gy * gy), 15) + EPS)
    # exact-flatness (constant fills): fraction of near-zero residual pixels
    F['flat_15'] = box((np.abs(rcb).max(2) < 0.75).astype(np.float32), 15)
    # white paper: natural highlights are clipped at 255, fills computed from the surroundings rarely are
    s255 = (page == 255).all(2).astype(np.float32)
    white = (page.min(2) >= 235).astype(np.float32)
    F['w255_15'] = box(s255, 15); F['w255_31'] = box(s255, 31)
    F['white_15'] = box(white, 15)
    F['w255_cs'] = F['w255_15'] - box(s255, 61)
    wmin = page.min(2).astype(np.float32)
    F['wmin_cs'] = box(wmin * white, 15) / (F['white_15'] + 1e-3) - box(wmin * white, 61) / (box(white, 61) + 1e-3)
    return F


CTX_KEYS = ['w255_15', 'cb_major_15', 'cb_minor_15', 'cb_ratio_15', 'cb_corr_15', 'g_corr_15',
            'hf_cr_15', 'hf_cb_15', 'hf_y_15', 'ac_x_15', 'ac_y_15', 'flat_15', 'g_minor_15', 'g_major_15']


def add_context(F, stamp_list, shape):
    """centre-surround and centre-minus-stamp-median context features."""
    H, W = shape
    sid = np.full((H, W), -1, np.int32)
    for k, (_, mk) in enumerate(stamp_list):
        sid[mk] = k
    for key in CTX_KEYS:
        a = F[key]
        F[key + '_cs61'] = a - box(a, 61)
        F[key + '_cs121'] = a - box(a, 121)
        z = np.zeros_like(a)
        for k, (_, mk) in enumerate(stamp_list):
            vals = a[mk]
            med = np.median(vals)
            mad = np.median(np.abs(vals - med)) + 1e-3
            z[mk] = (a[mk] - med) / mad
        F[key + '_zs'] = z
    return F, sid


def feature_stack(page, st):
    F = compute(page)
    F, sid = add_context(F, st, page.shape[:2])
    names = sorted(F.keys())
    return names, F, st, sid
