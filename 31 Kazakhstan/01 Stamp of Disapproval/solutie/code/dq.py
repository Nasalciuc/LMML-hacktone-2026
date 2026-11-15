"""Double-quantisation (DQ) forensics per stamp.

Each stamp image was JPEG-compressed once (quality ~82-92, its own 8x8 grid)
before it was edited and pasted on the page; the page was then saved at q97.
Unedited pixels keep DCT coefficients (on the stamp's grid) that sit on the
lattice of the first quantisation table.  Edited pixels (pasted digit, cloned
element, removed inscription, added noise) do not.

For every stamp component:
  1. grid offset (ox, oy) in 0..7: strongest periodic DCT histograms,
  2. first quality factor: standard libjpeg luminance table scaled by QF,
     fitted by maximum likelihood together with the noise sigma,
  3. per 8x8 block: log-likelihood ratio  log p(c | lattice) - log p(c | uniform)
     summed over informative coefficients (|c| > 0.6 q).
Touching stamps that came out of segmentation as one component carry two
different grids; a second grid is fitted on the large area the first grid
contradicts and every pixel takes the grid that explains its neighbourhood.
"""
import numpy as np
import cv2

ZZ = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2),
      (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3),
      (2, 4), (1, 5), (0, 6)]
T_LUM = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
CLIP_HI, CLIP_LO, CF_LLR = 254, 1, 0.05
_n = np.arange(8)
CM = np.sqrt(2 / 8) * np.cos((2 * _n[None, :] + 1) * _n[:, None] * np.pi / 16)
CM[0, :] = np.sqrt(1 / 8)
CM = CM.astype(np.float32)


def qtable(qf):
    s = 5000 / qf if qf < 50 else 200 - 2 * qf
    return np.maximum(np.floor((T_LUM * s + 50) / 100), 1)


def luma(img):
    f = img.astype(np.float32)
    return 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]


def block_dct(Y, ox, oy):
    H, W = Y.shape
    nby = (H - oy) // 8; nbx = (W - ox) // 8
    B = (Y[oy:oy + nby * 8, ox:ox + nbx * 8] - 128.0).reshape(nby, 8, nbx, 8).transpose(0, 2, 1, 3)
    return CM @ B @ CM.T


def block_frac(mask, ox, oy, nby, nbx):
    M = mask[oy:oy + nby * 8, ox:ox + nbx * 8].astype(np.float32)
    return M.reshape(nby, 8, nbx, 8).mean((1, 3))


def q_score(c, q):
    return float(np.cos(2 * np.pi * c / q).mean())


def find_grid(Y, region, clip, nfreq=6):
    best = (0.0, 0, 0)
    for oy in range(8):
        for ox in range(8):
            D = block_dct(Y, ox, oy)
            nby, nbx = D.shape[:2]
            use = (block_frac(region, ox, oy, nby, nbx) > 0.999) & (block_frac(clip, ox, oy, nby, nbx) < 0.02)
            tot = 0.0
            for (u, v) in ZZ[:nfreq]:
                c = D[use][:, u, v]
                c = c[np.abs(c) > 1.0]
                if len(c) < 50:
                    continue
                tot += max(q_score(c, q) for q in range(2, 12))
            if tot > best[0]:
                best = (tot, ox, oy)
    return best


def llr_coef(c, q, sg):
    d = c - q * np.round(c / q)
    f = np.zeros_like(d)
    for k in (-1, 0, 1):
        f += np.exp(-0.5 * ((d + k * q) / sg) ** 2)
    f /= (np.sqrt(2 * np.pi) * sg)
    return np.clip(np.log(q * f + 1e-6), -4, 4)


def fit_quality(D, use, nfreq=27, qfs=range(50, 100), sigmas=(0.5, 0.75, 1.0, 1.5)):
    coefs = {(u, v): D[use][:, u, v] for (u, v) in ZZ[:nfreq]}
    best = (-np.inf, 97, 1.0, 0)
    for qf in qfs:
        Q = qtable(qf)
        for sg in sigmas:
            tot = 0.0; n = 0
            for (u, v), c in coefs.items():
                q = Q[u, v]
                if q < 3 or q < 2.5 * sg:
                    continue
                ci = c[np.abs(c) > 0.6 * q]
                if len(ci) == 0:
                    continue
                tot += llr_coef(ci, q, sg).sum(); n += len(ci)
            if tot > best[0]:
                best = (tot, qf, sg, n)
    return best


def block_llr(D, inside, cf, qf, sg, nfreq=27, use_dc=True):
    nby, nbx = D.shape[:2]
    Q = qtable(qf)
    L = np.zeros((nby, nbx), np.float32); N = np.zeros((nby, nbx), np.float32)
    for (u, v) in ([(0, 0)] if use_dc else []) + ZZ[:nfreq]:
        q = Q[u, v]
        if q < 3 or q < 2.5 * sg:
            continue
        c = D[:, :, u, v]
        inf = (np.abs(c) > 0.6 * q) & inside & (cf < CF_LLR)
        L += np.where(inf, llr_coef(c, q, sg), 0); N += inf
    return L, N


def _grid_maps(Y, comp, clip, region, ox, oy):
    """fit QF on `region` with grid (ox,oy); return per-pixel L,N densities over the component"""
    H, W = Y.shape
    D = block_dct(Y, ox, oy)
    nby, nbx = D.shape[:2]
    use = (block_frac(region, ox, oy, nby, nbx) > 0.999) & (block_frac(clip, ox, oy, nby, nbx) < 0.02)
    fit = fit_quality(D, use)
    inside = block_frac(comp, ox, oy, nby, nbx) > 0.999
    cf = block_frac(clip, ox, oy, nby, nbx)
    L, N = block_llr(D, inside, cf, fit[1], fit[2])
    Lp = np.zeros((H, W), np.float32); Np = np.zeros((H, W), np.float32); Cp = np.zeros((H, W), np.float32)
    sl = (slice(oy, oy + nby * 8), slice(ox, ox + nbx * 8))
    Lp[sl] = np.kron(L / 64.0, np.ones((8, 8), np.float32))
    Np[sl] = np.kron(N / 64.0, np.ones((8, 8), np.float32))
    Cp[sl] = np.kron(cf * inside, np.ones((8, 8), np.float32))
    return Lp, Np, Cp, dict(qf=int(fit[1]), sigma=float(fit[2]), fit_llr=float(fit[0]), n=int(fit[3]))


def _density(Lp, Np, w):
    k = (w, w)
    return cv2.boxFilter(Lp, -1, k, normalize=False) / (cv2.boxFilter(Np, -1, k, normalize=False) + 5.0)


def analyse_component(Y, comp, clip, max_grids=3, min_area=12000):
    grids = []
    region = comp.copy()
    for g in range(max_grids):
        strength, ox, oy = find_grid(Y, region, clip)
        if strength < 0.5 or any((ox, oy) == (gg['ox'], gg['oy']) for gg in grids):
            break
        Lp, Np, Cp, inf = _grid_maps(Y, comp, clip, region, ox, oy)
        inf.update(ox=ox, oy=oy, strength=float(strength))
        grids.append(dict(L=Lp, N=Np, C=Cp, **inf))
        # area that no grid so far explains and that clearly contradicts them
        best = np.max(np.stack([_density(gg['L'], gg['N'], 49) for gg in grids]), 0)
        nwin = np.max(np.stack([cv2.boxFilter(gg['N'], -1, (49, 49), normalize=False) for gg in grids]), 0)
        bad = comp & (best < -0.1) & (nwin > 30)
        bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(bad)
        big = [k for k in range(1, n) if st[k, 4] > min_area]
        if not big:
            break
        region = np.isin(lab, big) & comp
    if not grids:
        z = np.zeros(Y.shape, np.float32)
        return z, z, z, [], np.full(Y.shape, -1, np.int8)
    if len(grids) == 1:
        g = grids[0]
        return g['L'], g['N'], g['C'], [_strip(g)], np.where(comp, 0, -1).astype(np.int8)
    dens = np.stack([_density(g['L'], g['N'], 61) for g in grids])
    choice = np.argmax(dens, 0)
    L = np.choose(choice, [g['L'] for g in grids]); N = np.choose(choice, [g['N'] for g in grids])
    C = np.choose(choice, [g['C'] for g in grids])
    gid = np.where(comp, choice, -1).astype(np.int8)
    return L, N, C, [_strip(g) for g in grids], gid


def _strip(g):
    return {k: v for k, v in g.items() if k not in ('L', 'N', 'C')}


def dq_maps(img, stamp_list, clip_hi=None, clip_lo=None, cf_llr=None):
    """per-pixel DQ maps for a page: L (LLR density), N (informative coefficients density),
    C (clipped fraction), plus stamp-level info per component."""
    H, W = img.shape[:2]
    Y = luma(img)
    global CF_LLR
    hi = CLIP_HI if clip_hi is None else clip_hi
    lo = CLIP_LO if clip_lo is None else clip_lo
    old_cf = CF_LLR
    if cf_llr is not None:
        CF_LLR = cf_llr
    clip = ((img >= hi) | (img <= lo)).any(2)
    Lp = np.zeros((H, W), np.float32); Np = np.zeros((H, W), np.float32); Cp = np.zeros((H, W), np.float32)
    S = np.zeros((H, W), np.float32); QF = np.zeros((H, W), np.float32); FL = np.zeros((H, W), np.float32)
    info = []
    for (x, y, w, h), mk in stamp_list:
        x0 = max(0, x - 8); y0 = max(0, y - 8); x1 = min(W, x + w + 8); y1 = min(H, y + h + 8)
        L, N, C, grids, gid = analyse_component(Y[y0:y1, x0:x1], mk[y0:y1, x0:x1], clip[y0:y1, x0:x1])
        sl = (slice(y0, y1), slice(x0, x1))
        m = mk[y0:y1, x0:x1]
        Lp[sl] += L * m; Np[sl] += N * m; Cp[sl] += C * m
        for k, g in enumerate(grids):
            sel = m & (gid == k)
            S[sl][sel] = g['strength']; QF[sl][sel] = g['qf']; FL[sl][sel] = g['fit_llr'] / max(g['n'], 1)
            g['ox_page'] = int((x0 + g['ox']) % 8); g['oy_page'] = int((y0 + g['oy']) % 8)
        info.append(grids)
    CF_LLR = old_cf
    return dict(L=Lp, N=Np, C=Cp, S=S, QF=QF, FL=FL), info


def dq_features(img, stamp_list, border=12, prefix='dq', clip_hi=None, clip_lo=None, cf_llr=None):
    M, info = dq_maps(img, stamp_list, clip_hi, clip_lo, cf_llr)
    L, N, C = M['L'], M['N'], M['C']
    # perforations / stamp edge are drawn after the first compression: ignore that band
    keep = np.zeros(L.shape, bool)
    for _, m in stamp_list:
        keep |= cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) > border
    L = L * keep; N = N * keep
    bs = lambda a, w: cv2.boxFilter(a, -1, (w, w), normalize=False, borderType=cv2.BORDER_CONSTANT)
    F = {}
    p = prefix
    for w in (17, 33, 49):
        F[f'{p}L{w}'] = bs(L, w)
        F[f'{p}N{w}'] = bs(N, w)
    F[f'{p}M33'] = F[f'{p}L33'] / (F[f'{p}N33'] + 3.0)
    F[f'{p}M17'] = F[f'{p}L17'] / (F[f'{p}N17'] + 3.0)
    F[f'{p}C33'] = bs(C, 33) / (33 * 33)
    H, W = L.shape
    z = np.zeros((H, W), np.float32); zr = np.zeros((H, W), np.float32)
    for _, m in stamp_list:
        v = F[f'{p}L33'][m]
        if len(v) == 0:
            continue
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1.0
        z[m] = (v - med) / mad
        zr[m] = np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    F[f'{p}Z33'] = z; F[f'{p}R33'] = zr
    F[f'{p}_grid'] = M['S']; F[f'{p}_qf'] = M['QF']; F[f'{p}_fit'] = M['FL']
    return F, M, info
