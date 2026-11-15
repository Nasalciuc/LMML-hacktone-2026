#!/usr/bin/env python3
"""The Azmari's Borrowed Melody - splice locator.

Usage:  python main.py test-1-input.wav   ->  writes detection.csv (start_s,end_s)

Idea: a few seconds taken from another take carry that take's own "fingerprint"
(infrasonic drift / DC, low-frequency rumble and intermodulation, hum, noise floor,
room decay / reverb, spectral envelope, waveform shape, dynamics, pitch content).
A frame inside the foreign segment therefore has no close match anywhere else in
the recording, while frames of the real take keep recurring.

  1. frame features (~46 ms hop), grouped by family
  2. per family: nearest-neighbour novelty with a temporal exclusion window
  3. candidate segments: peaks of the novelty curves + a blocked-CV LDA scan
  4. choose: most consistent candidate (CV AUC), then strongest novelty evidence;
     a clean unique step in the inaudible bands (infrasonic/LF, hum) takes priority
  5. extent: novelty re-measured only against frames outside the segment
  6. refine both boundaries on an LDA projection (epidemic change-point GLR)

Only numpy + scipy.  Always writes detection.csv (falls back to simpler logic on error).
"""
import sys
import csv
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import median_filter

try:
    from numpy.lib.stride_tricks import sliding_window_view as _swv
except ImportError:  # numpy < 1.20
    def _swv(a, w):
        from numpy.lib.stride_tricks import as_strided
        a = np.ascontiguousarray(a)
        return as_strided(a, shape=(a.shape[0] - w + 1, w), strides=(a.strides[0], a.strides[0]), writeable=False)

GROUPS = ["env", "ltas", "floor_rel", "floor_abs", "lf", "hum", "wave", "dyn", "chroma", "decay"]
SHORT_GROUPS = ["env", "floor_rel", "floor_abs", "wave", "chroma"]


# =============================================================================== I/O
def read_audio(path):
    try:
        sr, x = wavfile.read(path)
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        x = np.frombuffer(raw, dtype=dt)
        if nch > 1:
            x = x.reshape(-1, nch)
    x = np.asarray(x)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) * 256.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 65536.0
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64) * 32768.0
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.nan_to_num(x)
    return int(sr), x


def write_csv(start, end):
    with open("detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerow([f"{start:.3f}", f"{end:.3f}"])


# =============================================================================== features
def _finite(X):
    X = np.array(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            X[:, j] = 0.0
        elif bad.any():
            col[bad] = np.median(col[~bad])
            X[:, j] = col
    return X


def _decimate_to_1k(x, sr):
    q = max(int(round(sr / 1000.0)), 1)
    y = x
    rem = q
    for f in (2, 3, 5, 7, 11, 13):
        while rem % f == 0 and rem > 1:
            y = signal.decimate(y, f, ftype="fir", zero_phase=True)
            rem //= f
    if rem > 1:
        y = signal.resample_poly(y, 1, rem)
    return y, sr / q


def partial_decay_features(x, sr, centers, lags_s=(0.1, 0.25), win_s=1.5, rel_db=25.0):
    """Room signature: how fast strong partials die away (reverb tails slow the decay)."""
    win = 1 << int(np.round(np.log2(sr * 2048.0 / 22050.0)))
    hop = win // 8
    n = len(x)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect" if n > pad + 1 else "constant")
    m = 1 + (n - 1) // hop
    sw = _swv(xp, win)[::hop][:m]
    f = np.fft.rfftfreq(win, 1.0 / sr)
    band = (f >= 200) & (f < min(6000, sr / 2 - 1))
    hann = np.hanning(win)
    L = np.empty((m, int(band.sum())), dtype=np.float32)
    for s0 in range(0, m, 1024):
        Zc = np.fft.rfft(sw[s0:s0 + 1024] * hann, axis=1)[:, band]
        L[s0:s0 + 1024] = 10 * np.log10(Zc.real ** 2 + Zc.imag ** 2 + 1.0)
    pk = np.zeros(L.shape, dtype=bool)
    pk[:, 1:-1] = (L[:, 1:-1] > L[:, :-2]) & (L[:, 1:-1] >= L[:, 2:])
    pk &= L > L.max(axis=1, keepdims=True) - rel_db
    Lp = np.maximum(np.maximum(L[:, :-2], L[:, 1:-1]), L[:, 2:])
    W = max(int(round(win_s * sr / hop)), 3)
    cidx = np.clip(np.round(centers / hop).astype(int), 0, m - 1)
    cols = []
    for lag_s in lags_s:
        lag = max(int(round(lag_s * sr / hop)), 1)
        dlt = np.full(m, np.nan)
        for t0 in range(0, m - lag):
            b = np.where(pk[t0, 1:-1])[0]
            if len(b) < 2:
                continue
            dlt[t0] = np.percentile(Lp[t0 + lag, b] - L[t0, b + 1], 10)
        good = np.isfinite(dlt)
        if good.sum() == 0:
            cols += [np.zeros(len(centers)), np.zeros(len(centers))]
            continue
        dlt[~good] = np.nanmedian(dlt)
        S = _swv(np.pad(dlt, (W // 2, W - 1 - W // 2), mode="edge"), W)[cidx]
        cols.append(np.percentile(S, 5, axis=1))
        cols.append(np.percentile(S, 25, axis=1))
    return np.column_stack(cols)


def compute_features(x, sr):
    n = len(x)
    hop = max(int(round(sr * 1024.0 / 22050.0)), 32)
    win = 4 * hop
    nfr = 1 + (n - 1) // hop
    t = np.arange(nfr) * hop / sr
    centers = np.minimum(np.arange(nfr) * hop, n - 1)
    nyq = sr / 2.0

    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect" if n > pad + 1 else "constant")
    hann = np.hanning(win)
    f = np.fft.rfftfreq(win, 1.0 / sr)
    frames = _swv(xp, win)[::hop][:nfr]
    P = np.empty((nfr, len(f)))
    for s0 in range(0, nfr, 256):
        Zc = np.fft.rfft(frames[s0:s0 + 256] * hann, axis=1)
        P[s0:s0 + 256] = Zc.real ** 2 + Zc.imag ** 2
    P += 1.0
    logP = np.log(P)
    tot = np.log(P[:, f > 20].sum(1))

    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def win_mean_c(c, L):
        lo = np.clip(centers - L // 2, 0, n)
        hi = np.clip(centers + L // 2, 0, n)
        return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)

    def win_mean(y, L):
        return win_mean_c(np.concatenate([[0.0], np.cumsum(y)]), L)

    groups = {}

    # short-time spectral envelope (loudness normalised)
    edges = np.geomspace(25, nyq - 1, 41)
    E = np.zeros((nfr, len(edges) - 1))
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.sum() == 0:
            E[:, i] = logP[:, np.argmin(np.abs(f - edges[i]))]
        else:
            E[:, i] = np.log(P[:, m].mean(1))
    groups["env"] = E - tot[:, None]

    # long-term (2 s) spectrum: EQ / band limits / tilt
    Lw = max(int(round(2.0 * sr / hop)), 1)
    b_edges = np.geomspace(60, nyq - 1, 17)
    Pc = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    lo_i = np.clip(np.arange(nfr) - Lw // 2, 0, nfr)
    hi_i = np.clip(np.arange(nfr) + Lw // 2 + 1, 0, nfr)
    Pavg = (Pc[hi_i] - Pc[lo_i]) / (hi_i - lo_i)[:, None]
    del Pc
    totL = np.log(Pavg[:, f > 20].sum(1))
    LT = np.zeros((nfr, len(b_edges) - 1))
    for i in range(len(b_edges) - 1):
        m = (f >= b_edges[i]) & (f < b_edges[i + 1])
        if m.sum() == 0:
            m = np.argmin(np.abs(f - b_edges[i]))
            LT[:, i] = np.log(Pavg[:, m] + 1e-9)
        else:
            LT[:, i] = np.log(Pavg[:, m].mean(1) + 1e-9)
    groups["ltas"] = LT - totL[:, None]
    del Pavg

    # spectral floors (noise between partials)
    fl_edges = [e for e in [250, 500, 1000, 2000, 3000, 4500, 6000, 8000, 9500] if e < nyq - 200] + [nyq]
    FL = []
    for a_, b_ in zip(fl_edges[:-1], fl_edges[1:]):
        m = (f >= a_) & (f < b_)
        if m.sum() < 3:
            continue
        FL.append(np.percentile(logP[:, m], 10, axis=1))
    FL = np.array(FL).T
    groups["floor_rel"] = FL - tot[:, None]
    groups["floor_abs"] = FL

    # infrasonic drift / DC / low-frequency bands
    L = int(0.5 * sr)
    sos_lp = signal.butter(4, 20, "low", fs=sr, output="sos")
    lf = signal.sosfiltfilt(sos_lp, x)
    xe = win_mean_c(cs2, L) - win_mean_c(cs, L) ** 2
    cols = [win_mean_c(cs, L), win_mean_c(cs, 4 * L)]
    lf_m = win_mean(lf, L)
    lf_v = win_mean(lf * lf, L) - lf_m ** 2
    cols.append(0.5 * np.log(np.maximum(lf_v, 0) + 0.05))
    for (bl, bh) in [(20, 45), (45, 90), (90, 180)]:
        sos = signal.butter(4, [bl, bh], "band", fs=sr, output="sos")
        yb = signal.sosfiltfilt(sos, x)
        cols.append(0.5 * np.log(win_mean(yb * yb, L) + 0.05) - 0.5 * np.log(np.maximum(xe, 0) + 1.0))
    groups["lf"] = np.column_stack(cols)

    # mains hum: narrowband peakiness at 50/60 Hz and harmonics (~1 s windows)
    yd, fsd = _decimate_to_1k(x, sr)
    Lh = 1024
    ypad = np.pad(yd, (Lh // 2, Lh // 2), mode="reflect" if len(yd) > Lh // 2 + 1 else "constant")
    q = sr / fsd
    cd = np.clip(np.round(centers / q).astype(int), 0, len(yd) - 1)
    swv = _swv(ypad, Lh)
    fh = np.fft.rfftfreq(Lh, 1.0 / fsd)
    Ph = np.empty((nfr, len(fh)))
    hl = np.hanning(Lh)
    for s0 in range(0, nfr, 512):
        Zc = np.fft.rfft(swv[cd[s0:s0 + 512]] * hl, axis=1)
        Ph[s0:s0 + 512] = Zc.real ** 2 + Zc.imag ** 2
    Ph += 1e-3
    hum_cols = []
    for base in (50.0, 60.0):
        for hm in (1, 2, 3):
            fc = base * hm
            pk = (fh > fc - 1.5) & (fh < fc + 1.5)
            nb = ((fh > fc - 8) & (fh < fc - 3)) | ((fh > fc + 3) & (fh < fc + 8))
            if pk.sum() == 0 or nb.sum() == 0:
                continue
            hum_cols.append(np.log(Ph[:, pk].max(1)) - np.log(np.median(Ph[:, nb], axis=1)))
            hum_cols.append(np.log(Ph[:, pk].max(1)))
    groups["hum"] = np.column_stack(hum_cols) if hum_cols else np.zeros((nfr, 1))
    del Ph

    # waveform shape (100 ms)
    Ls = int(0.1 * sr)
    m1 = win_mean_c(cs, Ls)
    m2 = win_mean_c(cs2, Ls)
    m3 = win_mean(x ** 3, Ls)
    m4 = win_mean(x ** 4, Ls)
    var = np.maximum(m2 - m1 ** 2, 1e-6)
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2
    groups["wave"] = np.column_stack([np.clip(skew, -10, 10), np.log(np.clip(kurt, 1e-3, 1e3))])

    # band dynamics over 1 s (reverb tails, articulation)
    dyn = []
    Ld = max(int(round(1.0 * sr / hop)), 3)
    for a_, b_ in [(200, 800), (800, 2500), (2500, 6000)]:
        b_ = min(b_, nyq - 1)
        m = (f >= a_) & (f < b_)
        if m.sum() == 0:
            continue
        eb = np.log(P[:, m].sum(1))
        swe = _swv(np.pad(eb, (Ld // 2, Ld - 1 - Ld // 2), mode="edge"), Ld)
        dyn.append(np.percentile(swe, 90, axis=1) - np.percentile(swe, 10, axis=1))
        d1 = np.abs(np.diff(eb, prepend=eb[0]))
        dyn.append(np.convolve(d1, np.ones(Ld) / Ld, mode="same"))
    groups["dyn"] = np.column_stack(dyn) if dyn else np.zeros((nfr, 1))

    # pitch-class content
    m = (f > 60) & (f < min(2500, nyq - 1))
    midi = 69 + 12 * np.log2(f[m] / 440.0)
    pc = np.mod(np.round(midi), 12).astype(int)
    C = np.zeros((nfr, 12))
    Pm = P[:, m]
    for k in range(12):
        if (pc == k).any():
            C[:, k] = Pm[:, pc == k].sum(1)
    C = C / (C.sum(1, keepdims=True) + 1e-9)
    groups["chroma"] = np.log(C + 1e-3)

    groups["decay"] = partial_decay_features(x, sr, centers)

    for k in list(groups):
        groups[k] = _finite(groups[k])
    return t, groups


# =============================================================================== statistics
def robust_z(X, valid=None):
    X = np.asarray(X, dtype=np.float64)
    R = X if valid is None else X[valid]
    med = np.median(R, axis=0)
    mad = np.median(np.abs(R - med), axis=0) * 1.4826
    sd = R.std(axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
    return (X - med) / scale


def nn_novelty(Z, excl, k, valid):
    """Mean distance to the k nearest frames that are > excl frames away in time."""
    T = len(Z)
    sq = np.einsum("ij,ij->i", Z, Z)
    out = np.empty(T)
    idx = np.arange(T)
    kk = int(min(k, max(1, valid.sum() - 2 * excl - 2)))
    for s0 in range(0, T, 1024):
        s1 = min(T, s0 + 1024)
        G = Z[s0:s1] @ Z.T
        D = np.sqrt(np.maximum(sq[s0:s1, None] + sq[None, :] - 2 * G, 0))
        D[np.abs(idx[s0:s1, None] - idx[None, :]) <= excl] = np.inf
        D[:, ~valid] = np.inf
        part = np.partition(D, kk, axis=1)[:, :kk]
        part[~np.isfinite(part)] = np.nan
        with np.errstate(all="ignore"):
            out[s0:s1] = np.nanmean(part, axis=1)
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros(T)
    out[bad] = np.nanmedian(out[~bad])
    return out


def epi_all(y, lo, hi, mn, mx):
    yy = y[lo:hi]
    T = len(yy)
    yy = yy - yy.mean()
    sd = yy.std() + 1e-9
    c = np.concatenate([[0.0], np.cumsum(yy)])
    Ls = np.arange(mn, mx + 1)
    a = np.arange(T)[:, None]
    b = a + Ls[None, :]
    ok = b <= T
    bb = np.minimum(b, T)
    sin = c[bb] - c[a]
    nin = Ls[None, :].astype(float)
    nout = T - nin
    stat = sin * (1.0 / nin + 1.0 / np.maximum(nout, 1)) * np.sqrt(nin * np.maximum(nout, 0) / T) / sd
    stat[~ok] = -np.inf
    return stat, Ls


def top_segments(y, lo, hi, mn, mx, n=2):
    stat, Ls = epi_all(y, lo, hi, mn, mx)
    out = []
    A = np.arange(stat.shape[0])[:, None]
    B = A + Ls[None, :]
    for _ in range(n):
        kidx = int(np.argmax(stat))
        a, j = np.unravel_index(kidx, stat.shape)
        if not np.isfinite(stat[a, j]):
            break
        b = a + Ls[j]
        out.append((int(a + lo), int(b + lo)))
        stat[(np.minimum(B, b) - np.maximum(A, a)) > 0] = -np.inf
    return out


def scan_mean(y, lo, hi, mn, mx, gamma):
    c = np.concatenate([[0.0], np.cumsum(y)])
    best = (-np.inf, lo, lo + mn)
    for a in range(lo, hi - mn + 1):
        b = np.arange(a + mn, min(a + mx, hi) + 1)
        L = b - a
        val = (c[b] - c[a]) / L * L ** gamma
        j = int(np.argmax(val))
        if val[j] > best[0]:
            best = (val[j], a, int(b[j]))
    return best[1], best[2]


class RangeStats:
    """Prefix sums -> class means / scatter over unions of contiguous frame ranges."""

    def __init__(self, F):
        T, D = F.shape
        self.F = F
        self.c1 = np.vstack([np.zeros((1, D)), np.cumsum(F, axis=0)])
        self.c2 = np.concatenate([np.zeros((1, D, D)), np.cumsum(F[:, :, None] * F[:, None, :], axis=0)], axis=0)

    def stats(self, ranges):
        n, s1, s2 = 0, 0.0, 0.0
        for a, b in ranges:
            if b > a:
                n += b - a
                s1 = s1 + (self.c1[b] - self.c1[a])
                s2 = s2 + (self.c2[b] - self.c2[a])
        return n, s1, s2


def split_ranges(ranges, K):
    total = sum(b - a for a, b in ranges)
    bounds = [round(total * k / K) for k in range(K + 1)]
    out = [[] for _ in range(K)]
    pos = 0
    for a, b in ranges:
        L = b - a
        for k in range(K):
            lo = max(bounds[k], pos)
            hi = min(bounds[k + 1], pos + L)
            if hi > lo:
                out[k].append((a + lo - pos, a + hi - pos))
        pos += L
    return out


def auc_score(pin, pout):
    if len(pin) == 0 or len(pout) == 0:
        return 0.5
    allp = np.concatenate([pin, pout])
    r = np.argsort(np.argsort(allp)) + 1
    return (r[:len(pin)].sum() - len(pin) * (len(pin) + 1) / 2) / (len(pin) * len(pout))


def cv_auc(RS, a, b, lo, hi, guard, K=4, lam=0.3):
    """Blocked K-fold CV of a shrinkage LDA separating [a,b) from the rest."""
    inside = [(a, b)]
    outside = [(p, q) for p, q in [(lo, max(a - guard, lo)), (min(b + guard, hi), hi)] if q > p]
    in_ch = split_ranges(inside, K)
    out_ch = split_ranges(outside, K)
    F = RS.F
    D = F.shape[1]
    pins, pouts = [], []
    for k in range(K):
        tr_in = [r for j in range(K) if j != k for r in in_ch[j]]
        tr_out = [r for j in range(K) if j != k for r in out_ch[j]]
        n1, s1, q1 = RS.stats(tr_in)
        n0, s0, q0 = RS.stats(tr_out)
        if n1 < 3 or n0 < 3 or not in_ch[k] or not out_ch[k]:
            continue
        m1, m0 = s1 / n1, s0 / n0
        Sw = (q1 - n1 * np.outer(m1, m1) + q0 - n0 * np.outer(m0, m0)) / (n1 + n0 - 2)
        Sw = (1 - lam) * Sw + lam * np.diag(np.diag(Sw)) + 1e-6 * np.eye(D)
        try:
            w = np.linalg.solve(Sw, m1 - m0)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Sw, m1 - m0, rcond=None)[0]
        c = 0.5 * (m1 + m0) @ w
        pins.append(np.concatenate([F[p:q] for p, q in in_ch[k]]) @ w - c)
        pouts.append(np.concatenate([F[p:q] for p, q in out_ch[k]]) @ w - c)
    if not pins:
        return 0.5
    return float(auc_score(np.concatenate(pins), np.concatenate(pouts)))


def lda_projection(F, a, b, lo, hi, guard, lam=0.3):
    T, D = F.shape
    m = np.zeros(T, bool)
    m[a:b] = True
    o = np.zeros(T, bool)
    o[lo:hi] = True
    o[max(a - guard, 0):min(b + guard, T)] = False
    X1, X0 = F[m], F[o]
    if len(X1) < 3 or len(X0) < 3:
        return np.zeros(T)
    m1, m0 = X1.mean(0), X0.mean(0)
    Sw = (np.cov(X1.T) * (len(X1) - 1) + np.cov(X0.T) * (len(X0) - 1)) / (len(X1) + len(X0) - 2)
    Sw = np.atleast_2d(Sw)
    Sw = (1 - lam) * Sw + lam * np.diag(np.diag(Sw)) + 1e-6 * np.eye(D)
    try:
        w = np.linalg.solve(Sw, m1 - m0)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Sw, m1 - m0, rcond=None)[0]
    p = F @ w
    med = np.median(p[o])
    mad = np.median(np.abs(p[o] - med)) * 1.4826 + 1e-9
    return (p - med) / mad


def glr_local(y, a, b, lo, hi, rad, mn):
    """Epidemic change-point GLR with both boundaries within +-rad of (a, b)."""
    yy = np.clip(y[lo:hi], -20, 20)
    T = len(yy)
    yy = yy - np.median(yy)
    c = np.concatenate([[0.0], np.cumsum(yy)])
    tot = c[-1]
    best = (-np.inf, a, b)
    for aa in range(max(lo, a - rad), min(hi - mn, a + rad) + 1):
        bb = np.arange(max(aa + mn, b - rad), min(hi, b + rad) + 1)
        if len(bb) == 0:
            continue
        nin = (bb - aa).astype(float)
        nout = T - nin
        sin = c[bb - lo] - c[aa - lo]
        stat = (sin / nin - (tot - sin) / np.maximum(nout, 1)) * np.sqrt(nin * np.maximum(nout, 0) / T)
        j = int(np.argmax(stat))
        if stat[j] > best[0]:
            best = (stat[j], aa, int(bb[j]))
    return best[1], best[2]


OVERRIDE_GROUPS = ("lf", "hum")
OVERRIDE_ALWAYS = True


def background_override(raw_curves, lo, hi, mn, mx, groups=None, C=6.0, M=8.0):
    groups = OVERRIDE_GROUPS if groups is None else groups
    best = None
    for g in groups:
        raw = raw_curves[g]
        stat, Ls = epi_all(np.clip(raw, -3, 30), lo, hi, mn, mx)
        k1 = int(np.argmax(stat))
        a1, j1 = np.unravel_index(k1, stat.shape)
        s1 = stat[a1, j1]
        if not np.isfinite(s1):
            continue
        b1 = a1 + Ls[j1]
        A_ = np.arange(stat.shape[0])[:, None]
        B_ = A_ + Ls[None, :]
        st2 = stat.copy()
        st2[(np.minimum(B_, b1) - np.maximum(A_, a1)) > 0] = -np.inf
        s2 = np.max(st2)
        c = s1 / max(s2, 1e-3) if np.isfinite(s2) else np.inf
        a, b = int(a1 + lo), int(b1 + lo)
        med = float(np.median(raw[a:b]))
        if c >= C and med >= M:
            score = min(c, 1e6) * med
            if best is None or score > best[0]:
                best = (score, a, b)
    return None if best is None else (best[1], best[2])


def segment_novelty_refine(Zs, valid, a, b, lo, hi, mn, fps, k=5, self_excl_s=2.0, clipv=8.0):
    T = len(valid)
    L = b - a
    W = int(max(4.0 * fps, 0.5 * L))
    R = valid.copy()
    R[max(a - W, 0):min(b + W, T)] = False
    se = int(self_excl_s * fps)
    if R.sum() < max(4 * fps, 2 * se + k + 2):
        return a, b
    idx = np.arange(T)
    sm = max(int(0.5 * fps) | 1, 1)
    m_in = np.zeros(T, bool)
    m_in[a:b] = True
    num = np.zeros(T)
    wsum = 0.0
    for g in GROUPS:
        Z = Zs[g]
        sq = np.einsum("ij,ij->i", Z, Z)
        nv = np.empty(T)
        kk = int(min(k, R.sum() - 2 * se - 1))
        Rz = Z[R]
        Ridx = idx[R]
        for s0 in range(0, T, 1024):
            s1 = min(T, s0 + 1024)
            Dm = np.sqrt(np.maximum(sq[s0:s1, None] + sq[R][None, :] - 2 * Z[s0:s1] @ Rz.T, 0))
            Dm[np.abs(idx[s0:s1, None] - Ridx[None, :]) <= se] = np.inf
            part = np.partition(Dm, kk, axis=1)[:, :kk]
            part[~np.isfinite(part)] = np.nan
            with np.errstate(all="ignore"):
                nv[s0:s1] = np.nanmean(part, axis=1)
        bad = ~np.isfinite(nv)
        if bad.all():
            continue
        nv[bad] = np.nanmedian(nv[~bad])
        med = np.median(nv[R])
        mad = np.median(np.abs(nv[R] - med)) * 1.4826 + 1e-9
        z = median_filter((nv - med) / mad, size=sm, mode="nearest")
        z[~valid] = 0.0
        z = np.clip(z, -3, clipv)
        w = max(z[m_in].mean() - z[R].mean(), 0.0) ** 2
        num += w * z
        wsum += w
    if wsum <= 0:
        return a, b
    return glr_local(num / wsum, a, b, lo, hi, W, mn)


# =============================================================================== detector
def detect(x, sr):
    k, excl_s, smooth_s, edge_s = 5, 10.0, 0.5, 1.0
    min_s, max_s, clipv = 1.5, 16.0, 8.0
    scan_step_s, auc_tol, rad_s, n_iter, short_auc = 0.5, 0.02, 3.0, 3, 0.97

    t, G = compute_features(x, sr)
    T = len(t)
    fps = 1.0 / (t[1] - t[0])
    dur = len(x) / sr
    edge = int(edge_s * fps)
    if T - 2 * edge < int(4 * fps):
        edge = max(int(0.1 * T), 1)
    lo, hi = edge, T - edge
    valid = np.zeros(T, bool)
    valid[lo:hi] = True
    excl = int(min(excl_s, dur / 4) * fps)
    sm = max(int(smooth_s * fps) | 1, 1)
    mn = max(int(min_s * fps), 2)
    mx = max(min(int(max_s * fps), int(0.6 * (hi - lo))), mn + 1)
    guard = int(0.5 * fps)

    # 1-2. features and per-family novelty curves
    Zs, curves, raw_curves = {}, {}, {}
    for g in GROUPS:
        Z = np.clip(robust_z(G[g], valid), -8, 8)
        Zs[g] = Z
        nv = nn_novelty(Z, excl, k, valid)
        z = robust_z(nv[:, None], valid)[:, 0]
        z = median_filter(z, size=sm, mode="nearest")
        z[~valid] = 0.0
        raw_curves[g] = z
        curves[g] = np.clip(z, -3, clipv)
    comb = np.mean([curves[g] for g in GROUPS], axis=0)

    # 3. candidates
    cands = set()
    for g in GROUPS:
        for a, b in top_segments(curves[g], lo, hi, mn, mx, n=2):
            cands.add((a, b))
    for a, b in top_segments(comb, lo, hi, mn, mx, n=2):
        cands.add((a, b))
    for gm in (0.25, 0.5):
        cands.add(scan_mean(comb - np.median(comb[valid]), lo, hi, mn, mx, gm))

    F = np.hstack([Zs[g] for g in GROUPS] + [curves[g][:, None] for g in GROUPS])
    dec = max(2, int(np.ceil(T / 700.0)))
    Fd = F[::dec]
    RSd = RangeStats(Fd)
    lod, hid = (lo + dec - 1) // dec, hi // dec
    guardd = max(guard // dec, 1)
    step = max(int(round(scan_step_s * fps / dec)), 1)
    mnd, mxd = max(mn // dec, 2), max(mx // dec, 3)
    res = []
    for a in range(lod, hid - mnd + 1, step):
        for L in range(mnd, mxd + 1, step):
            b = a + L
            if b > hid:
                break
            res.append((cv_auc(RSd, a, b, lod, hid, guardd), a, b))
    res.sort(key=lambda r: -r[0])
    picked = []
    for r in res:
        if all(min(r[2], q[2]) - max(r[1], q[1]) <= 0 for q in picked):
            picked.append(r)
        if len(picked) >= 4:
            break
    for _, a, b in picked:
        cands.add((a * dec, min(b * dec, hi)))

    # 4. choose: near-best consistency, then strongest novelty evidence
    sds = {g: curves[g][valid].std() + 1e-9 for g in GROUPS}
    scored = []
    for (a, b) in cands:
        if b - a < 2:
            continue
        m_in = np.zeros(T, bool)
        m_in[a:b] = True
        m_out = valid.copy()
        m_out[max(a - guard, 0):min(b + guard, T)] = False
        nin, nout = m_in.sum(), m_out.sum()
        if nout < 2:
            continue
        E = 0.0
        for g in GROUPS:
            d = (curves[g][m_in].mean() - curves[g][m_out].mean()) / sds[g]
            E += max(d * np.sqrt(nin * nout / (nin + nout)), 0.0) ** 2
        A = cv_auc(RSd, a // dec, max(b // dec, a // dec + 2), lod, hid, guardd)
        scored.append((A, E, a, b))
    maxA = max(s[0] for s in scored)
    pool = [s for s in scored if s[0] >= maxA - auc_tol]
    _, _, a, b = max(pool, key=lambda s: s[1])

    # 4b. inaudible-band override: a clean, unique step in the infrasonic/LF or hum
    #     fingerprint is the most direct evidence of a second take
    ov = background_override(raw_curves, lo, hi, mn, mx)
    if ov is not None:
        a0, b0 = ov
        if OVERRIDE_ALWAYS or min(b, b0) - max(a, a0) <= 0:
            a, b = a0, b0

    # 4c. extent: novelty of every frame measured only against frames outside the segment
    a, b = segment_novelty_refine(Zs, valid, a, b, lo, hi, mn, fps, k)

    # 5. boundary refinement
    rad = int(rad_s * fps)
    F_short = np.hstack([Zs[g] for g in SHORT_GROUPS])
    F_raw = np.hstack([Zs[g] for g in GROUPS])
    for _ in range(n_iter):
        ps = lda_projection(F_short, a, b, lo, hi, guard)
        m_in = np.zeros(T, bool)
        m_in[a:b] = True
        m_out = valid.copy()
        m_out[max(a - guard, 0):min(b + guard, T)] = False
        if auc_score(ps[m_in], ps[m_out]) >= short_auc:
            p = ps
        else:
            p = lda_projection(F_raw, a, b, lo, hi, guard)
        a2, b2 = glr_local(p, a, b, lo, hi, rad, mn)
        if (a2, b2) == (a, b):
            break
        a, b = a2, b2

    start = float(t[a])
    end = float(t[min(b - 1, T - 1)] + 0.5 / fps)
    return start, end


def fallback(x, sr):
    """Coarse backup: equal-weight novelty on a few cheap features."""
    t, G = compute_features(x, sr)
    T = len(t)
    fps = 1.0 / (t[1] - t[0])
    valid = np.ones(T, bool)
    excl = int(min(10.0, len(x) / sr / 4) * fps)
    zs = []
    for g in ("env", "lf", "chroma"):
        Z = np.clip(robust_z(G[g]), -8, 8)
        zs.append(np.clip(robust_z(nn_novelty(Z, excl, 5, valid)[:, None])[:, 0], -3, 8))
    comb = np.mean(zs, axis=0)
    mn = max(int(1.5 * fps), 2)
    mx = max(min(int(16 * fps), int(0.6 * T)), mn + 1)
    a, b = top_segments(comb, 0, T, mn, mx, n=1)[0]
    return float(t[a]), float(t[min(b - 1, T - 1)])


def main():
    if len(sys.argv) < 2:
        print("usage: python main.py <input.wav>")
        sys.exit(1)
    path = sys.argv[1]
    start, end = None, None
    try:
        sr, x = read_audio(path)
        dur = len(x) / float(sr)
        try:
            start, end = detect(x, sr)
        except Exception as exc:  # keep going with a simpler method
            print("detector failed:", repr(exc), file=sys.stderr)
            try:
                start, end = fallback(x, sr)
            except Exception as exc2:
                print("fallback failed:", repr(exc2), file=sys.stderr)
                start, end = 0.4 * dur, 0.5 * dur
        start = max(0.0, min(start, dur))
        end = max(start, min(end, dur))
    except Exception as exc:
        print("could not read input:", repr(exc), file=sys.stderr)
        start, end = 0.0, 0.0
    write_csv(start, end)
    print(f"start_s={start:.3f} end_s={end:.3f}")


if __name__ == "__main__":
    main()
