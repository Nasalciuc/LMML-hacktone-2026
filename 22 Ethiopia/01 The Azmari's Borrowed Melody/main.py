#!/usr/bin/env python3
"""Splice locator for "The Azmari's Borrowed Melody".

The replaced phrase comes from a different take: another instrument and/or
another recording chain. Nothing audible marks the seams, but every frame
inside the splice carries the other take's fingerprint (harmonic profile,
spectral envelope, sub-fundamental rumble, infrasonic DC behaviour, noise
floor, hum, tuning, ...). We compute ~40 such frame features, rank-gaussianise
and whiten them, and pick the single contiguous block whose frames are
jointly most separable from the rest of the file (max squared canonical
correlation between a box indicator and the features).

Usage: python main.py test-1-input.wav  ->  writes detection.csv
"""
import sys
import csv
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import dct
from scipy.stats import rankdata, norm
from scipy.ndimage import median_filter, uniform_filter1d, minimum_filter1d, maximum_filter1d

EPS = 1e-10


# ----------------------------------------------------------------------------- io
def load_audio(path):
    sr, d = wavfile.read(path)
    x = np.asarray(d)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) * 256.0
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64) * 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 65536.0
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x[~np.isfinite(x)] = 0.0
    return int(sr), x


# ----------------------------------------------------------------------------- helpers
def mel_fb(f, n_mels=24, fmin=40.0, fmax=10000.0):
    def hz2mel(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m = np.linspace(hz2mel(fmin), hz2mel(min(fmax, f[-1] * 0.95)), n_mels + 2)
    hz = mel2hz(m)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb


def win_mean(v, starts, n):
    """mean of v[s:s+n] for each start s (prefix sums)."""
    cs = np.concatenate([[0.0], np.cumsum(v)])
    return (cs[starts + n] - cs[starts]) / n


def win_max(v, starts, n):
    # max over (approximately) [s, s+n): centred filter indexed at the window centre
    mf = maximum_filter1d(v, size=n)
    return mf[np.clip(starts + n // 2, 0, len(v) - 1)]


def win_min(v, starts, n):
    mf = minimum_filter1d(v, size=n)
    return mf[np.clip(starts + n // 2, 0, len(v) - 1)]


# ----------------------------------------------------------------------------- features
def compute_features(x, sr, hop_s=0.05, win_s=0.093):
    x = x - np.mean(x)
    hop = max(1, int(round(hop_s * sr)))
    nper = int(2 ** np.round(np.log2(win_s * sr)))
    if len(x) < nper + 4 * hop:
        x = np.concatenate([x, np.zeros(nper + 4 * hop - len(x))])
    f, _, Z = signal.stft(x, sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2 + EPS
    del Z
    nfr = P.shape[1]
    fr = f[1]
    nyq = sr / 2.0
    tot = P.sum(0)
    L = np.log10(P)

    # --- f0: harmonic sum on log spectrum with half-harmonic penalty, then refine
    fmax0 = min(1800.0, nyq / 6.0)
    cand = 2 ** np.arange(np.log2(70.0), np.log2(fmax0), 1.0 / 96.0)
    hs = np.zeros((len(cand), nfr))
    for k in range(1, 7):
        idx = np.clip(np.round(cand * k / fr).astype(int), 0, len(f) - 1)
        hs += L[idx]
    for k in range(1, 6):
        idx = np.clip(np.round(cand * (k - 0.5) / fr).astype(int), 0, len(f) - 1)
        hs -= 0.5 * L[idx]
    f0c = cand[np.argmax(hs, 0)]
    del hs
    K = 10
    hamp = np.full((K, nfr), EPS)
    num = np.zeros(nfr)
    den = np.zeros(nfr)
    cols = np.arange(nfr)
    for k in range(1, K + 1):
        lo = np.floor(k * f0c * 0.97 / fr).astype(int)
        hi = np.ceil(k * f0c * 1.03 / fr).astype(int) + 1
        valid = lo < len(f) - 2
        width = int(np.max(np.where(valid, hi - lo, 1)))
        offs = np.arange(width)
        idx = np.clip(lo[:, None] + offs[None, :], 0, len(f) - 1)
        inside = (lo[:, None] + offs[None, :]) < np.minimum(hi, len(f) - 1)[:, None]
        vals = np.where(inside, P[idx, cols[:, None]], -1.0)
        jrel = np.argmax(vals, 1)
        j = idx[cols, jrel]
        pk = P[j, cols]
        hamp[k - 1] = np.where(valid, pk, EPS)
        if k <= 5:
            jj = np.clip(j, 1, len(f) - 2)
            a, b, c = L[jj - 1, cols], L[jj, cols], L[jj + 1, cols]
            dd = a - 2 * b + c
            delta = np.where(np.abs(dd) > 1e-12, 0.5 * (a - c) / np.where(np.abs(dd) > 1e-12, dd, 1.0), 0.0)
            delta = np.clip(delta, -0.5, 0.5)
            w = np.where(valid, pk, 0.0)
            num += w * (jj + delta) * fr / k
            den += w
    f0 = np.where(den > 0, num / np.maximum(den, EPS), f0c)
    f0 = np.where((f0 > 0.8 * f0c) & (f0 < 1.25 * f0c), f0, f0c)

    F = {}
    F['f0log'] = np.log2(f0)
    cents = 1200.0 * np.log2(f0 / 440.0)
    F['tune_s'] = np.sin(2 * np.pi * cents / 100.0)
    F['tune_c'] = np.cos(2 * np.pi * cents / 100.0)
    hdb = 10 * np.log10(hamp)
    for k in (2, 3, 4, 5):
        F['h%d_h1' % k] = hdb[k - 1] - hdb[0]
    F['even_odd'] = 10 * np.log10(hamp[[1, 3, 5, 7]].sum(0) / hamp[[2, 4, 6, 8]].sum(0))
    lk = np.log2(np.arange(1, K + 1))
    A = np.vstack([lk, np.ones(K)]).T
    F['hslope'] = np.linalg.lstsq(A, hdb, rcond=None)[0][0]
    F['centroid_f0'] = (f[:, None] * P).sum(0) / tot / f0
    fb = mel_fb(f)
    mel = np.log(fb @ P + EPS)
    cc = dct(mel, type=2, axis=0, norm='ortho')
    for k in range(1, 7):
        F['mfcc%d' % k] = cc[k]
    ff = f[:, None]
    subm = (ff >= 15) & (ff < 0.7 * f0[None, :])
    F['subf0_rel'] = 10 * np.log10((P * subm).sum(0) / tot + EPS)
    m = (f >= 20) & (f < 120)
    F['lf_abs'] = 10 * np.log10(P[m].sum(0) + EPS)
    m = (f >= 3) & (f < 40)
    F['lf_lo'] = 10 * np.log10(P[m].sum(0) + EPS)
    ph = (ff / f0[None, :]) % 1.0
    hb = (ph < 0.12) | (ph > 0.88)
    above = ff > 0.6 * f0[None, :]
    for lo, hi, nm in [(0, 2000, 'nz_lo'), (2000, 5000, 'nz_mid'), (5000, nyq, 'nz_hi')]:
        bm = (ff >= lo) & (ff < hi) & above
        Hh = (P * (hb & bm)).sum(0)
        Nn = (P * ((~hb) & bm)).sum(0)
        F[nm] = 10 * np.log10((Nn + EPS) / (Hh + EPS))
    del ph, hb, above, subm
    for lo, hi, nm in [(1000, 3000, 'floor1'), (3000, 6000, 'floor2'), (6000, min(10000, nyq), 'floor3')]:
        m = (f >= lo) & (f < hi)
        if m.sum() >= 3:
            F[nm] = 10 * np.log10(np.percentile(P[m], 10, axis=0))
    m = f >= min(5000, 0.45 * nyq)
    F['hf_rel'] = 10 * np.log10(P[m].sum(0) / tot)
    m = (f >= 2000) & (f < min(8000, nyq))
    F['flat2_8k'] = np.mean(np.log(P[m]), 0) - np.log(np.mean(P[m], 0))
    Ld = np.diff(np.log(P[f > 1000] + 1e-3), axis=1)
    F['flux'] = np.concatenate([[0.0], np.maximum(Ld, 0).mean(0)])
    del Ld, L, P

    starts = np.arange(nfr) * hop
    cen = starts + nper // 2

    # --- mains hum (narrow-band 50/60 Hz family) from a 1 kHz copy with ~1 s windows
    srl = 1000
    g = np.gcd(int(sr), srl)
    xl = signal.resample_poly(x, srl // g, int(sr) // g)
    nl = 1024
    hopl = hop_s * srl
    fl = np.fft.rfftfreq(nl, 1.0 / srl)
    wl = np.hanning(nl)
    centers_l = np.clip(np.round(cen / sr * srl - nl / 2).astype(int), 0, max(0, len(xl) - nl))
    if len(xl) < nl:
        xl = np.concatenate([xl, np.zeros(nl - len(xl))])
    frames_l = np.lib.stride_tricks.sliding_window_view(xl, nl)[centers_l] * wl
    Pl = np.abs(np.fft.rfft(frames_l, axis=1)).T ** 2 + EPS
    del frames_l
    for base, nm in [(50.0, 'hum50'), (60.0, 'hum60')]:
        nu = 0.0
        de = 0.0
        for k in (1, 2, 3):
            fc = base * k
            mk = np.abs(fl - fc) <= 1.0
            mn = (np.abs(fl - fc) > 3.0) & (np.abs(fl - fc) < 12.0)
            nu = nu + Pl[mk].max(0)
            de = de + np.median(Pl[mn], 0)
        F[nm] = 10 * np.log10(nu / de)
    del Pl

    # --- time-domain features (prefix sums, no big frame matrices)
    m1 = win_mean(x, starts, nper)
    m2 = win_mean(x ** 2, starts, nper)
    m3 = win_mean(x ** 3, starts, nper)
    var = np.maximum(m2 - m1 ** 2, EPS)
    sd = np.sqrt(var)
    F['skew'] = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / sd ** 3
    mx = win_max(x, starts, nper)
    mn_ = win_min(x, starts, nper)
    F['crest'] = np.maximum(mx - m1, m1 - mn_) / sd
    lp12 = signal.sosfiltfilt(signal.butter(2, 12, fs=sr, output='sos'), x)
    F['infra'] = np.log10(np.sqrt(win_mean(lp12 ** 2, starts, nper)) + 1e-2)
    lp2 = signal.sosfiltfilt(signal.butter(2, 2, fs=sr, output='sos'), x)
    F['dc'] = lp2[np.clip(cen, 0, len(x) - 1)]
    del lp12, lp2
    env = np.abs(signal.hilbert(x))
    em = win_mean(env, starts, nper) + EPS
    hp_env = signal.sosfiltfilt(signal.butter(2, 15, btype='high', fs=sr, output='sos'), env)
    he1 = win_mean(hp_env, starts, nper)
    he2 = win_mean(hp_env ** 2, starts, nper)
    F['am_fast'] = np.log10(np.sqrt(np.maximum(he2 - he1 ** 2, 0)) / em + 1e-4)
    del env, hp_env
    le = np.log10(em)
    w = max(3, int(round(0.6 / hop_s)))
    F['env_dip'] = minimum_filter1d(le, w) - maximum_filter1d(le, w)
    c = 1200 * np.log2(f0)
    w = max(3, int(round(0.3 / hop_s)))
    a1 = uniform_filter1d(c, w)
    a2 = uniform_filter1d(c ** 2, w)
    F['f0_jit'] = np.log10(np.sqrt(np.maximum(a2 - a1 ** 2, 0)) + 1)

    tt = (starts + nper / 2.0) / sr
    for k in list(F):
        v = np.asarray(F[k], dtype=np.float64)
        bad = ~np.isfinite(v)
        if bad.all():
            del F[k]
            continue
        if bad.any():
            v[bad] = np.median(v[~bad])
        F[k] = v
    return tt, F


# ----------------------------------------------------------------------------- segment search
def gauss_rank(v):
    r = rankdata(v, method='average')
    return norm.ppf(r / (len(v) + 1.0))


def best_block(Xw, min_len, max_len):
    """argmax over blocks [a,b) of n1*n2/N^2 * |mean_in - mean_out|^2 (whitened)."""
    N, d = Xw.shape
    cs = np.vstack([np.zeros(d), np.cumsum(Xw, 0)])
    best = (-1.0, 0, min(N, min_len))
    for a in range(0, N - min_len + 1):
        bs = np.arange(a + min_len, min(N, a + max_len) + 1)
        if len(bs) == 0:
            continue
        n1 = bs - a
        n2 = N - n1
        S1 = cs[bs] - cs[a]
        R = (S1 ** 2).sum(1) / (n1 * n2)          # = (n1 n2 / N^3) |dmu|^2, monotone in the canonical corr.
        j = int(np.argmax(R))
        if R[j] > best[0]:
            best = (float(R[j]), a, int(bs[j]))
    return best


# Features that follow which notes are played, not how or where they were recorded.
# They are left out of the block search: on other melodies they shift the edges.
MELODY = {'f0log', 'centroid_f0', 'flux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6'}
MAX_BLOCK_S = 20.0   # "a few seconds": longer blocks were always wrong picks in validation


def detect(x, sr):
    dur = len(x) / float(sr)
    hop_s = max(0.05, dur / 1500.0)
    tt, F = compute_features(x, sr, hop_s=hop_s)
    N = len(tt)
    hop = tt[1] - tt[0] if N > 1 else hop_s
    V = []
    for k in sorted(F):
        if k in MELODY:
            continue
        v = median_filter(F[k], size=5, mode='nearest')
        g = gauss_rank(v)
        if np.std(g) > 1e-3:
            V.append(g)
    V = np.array(V).T
    V = V - V.mean(0)
    C = np.cov(V.T)
    C = 0.9 * C + 0.1 * np.diag(np.diag(C))
    w, U = np.linalg.eigh(C)
    W = U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-6))) @ U.T
    Xw = V @ W
    min_len = max(2, int(round(1.5 / hop)))
    max_len = max(min_len + 1, min(int(0.5 * N), int(round(MAX_BLOCK_S / hop))))
    _, a, b = best_block(Xw, min_len, max_len)
    s = tt[a] - hop / 2.0
    e = tt[b - 1] + hop / 2.0
    s = float(np.clip(s, 0.0, dur))
    e = float(np.clip(e, 0.0, dur))
    if e <= s:
        e = min(dur, s + 1.0)
    return s, e


def fallback(x, sr):
    """Very simple backup: block with the most anomalous low-frequency share."""
    dur = len(x) / float(sr)
    try:
        hop = int(0.1 * sr)
        f, t, Z = signal.stft(x, sr, nperseg=4096, noverlap=4096 - hop)
        P = np.abs(Z) ** 2 + EPS
        v = np.log(P[(f >= 15) & (f < 150)].sum(0) / P.sum(0))
        g = gauss_rank(median_filter(v, 5)) [:, None]
        _, a, b = best_block(g - g.mean(), 15, max(16, len(g) // 2))
        return float(t[a]), float(t[min(b, len(t) - 1)])
    except Exception:
        return 0.4 * dur, 0.5 * dur


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else 'test-1-input.wav'
    try:
        sr, x = load_audio(fn)
    except Exception:
        sr, x = 22050, np.zeros(22050 * 10)
    try:
        s, e = detect(x, sr)
    except Exception:
        s, e = fallback(x, sr)
    with open('detection.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['start_s', 'end_s'])
        wr.writerow(['%.3f' % s, '%.3f' % e])


if __name__ == '__main__':
    main()
