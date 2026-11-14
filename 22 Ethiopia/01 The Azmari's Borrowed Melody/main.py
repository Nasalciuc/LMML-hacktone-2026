"""Find the replaced stretch in a recording.

    python main.py test-1-input.wav      -> detection.csv (start_s,end_s)

A pasted-in phrase comes from a different take, so the recording chain
around it is different even when the seam is inaudible: very-low-frequency
drift and DC, rumble below the instrument, the hiss floor between notes,
hum lines, reverb (how sharp note changes and dips are), and the overall
spectral colour. Each is tracked on 0.4 s frames every 0.1 s; the tracks
are decorrelated and the stretch whose mean differs most from the rest of
the file (Hotelling-style summed squared t) is reported.
"""
import csv
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

HOP = 0.1
WIN = 0.4
MIN_LEN = 1.5
MAX_LEN = 20.0


def _frames(y, sr, win=WIN, hop=HOP):
    h, w = int(hop * sr), int(win * sr)
    n = max(1, (len(y) - w) // h + 1)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return y[idx]


def _band(x, sr, lo, hi):
    if lo <= 0:
        sos = signal.butter(4, hi, "lp", fs=sr, output="sos")
    elif hi >= sr / 2:
        sos = signal.butter(4, lo, "hp", fs=sr, output="sos")
    else:
        sos = signal.butter(4, [lo, hi], "bp", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def _logrms(fr):
    return np.log(fr.std(1) + 1e-6)


def _mel_edges(lo, hi, n):
    m = np.linspace(2595 * np.log10(1 + lo / 700), 2595 * np.log10(1 + hi / 700), n + 2)
    return 700 * (10 ** (m / 2595) - 1)


def features(x, sr, hop=HOP, win=WIN):
    x = x.astype(float)
    fr = _frames(x, sr, win, hop)
    n = len(fr)
    tracks = {}

    tracks["dc"] = fr.mean(1)
    tracks["vlf"] = _logrms(_frames(_band(x, sr, 0, 8), sr, win, hop))
    tracks["rumble"] = _logrms(_frames(_band(x, sr, 20, 70), sr, win, hop))

    s = max(1, int(0.01 * sr))
    k = fr.shape[1] // s
    sub = fr[:, : k * s].reshape(n, k, s).std(2)
    tracks["floor"] = np.log(np.percentile(sub, 5, axis=1) + 1e-6)
    lsub = np.log(sub + 1e-6)
    tracks["env_std"] = lsub.std(1)
    tracks["dip"] = np.percentile(lsub, 5, axis=1) - np.median(lsub, axis=1)
    if sr / 2 > 7000:
        hf = _frames(_band(x, sr, 6000, sr / 2 - 1), sr, win, hop)
        sub = hf[:, : k * s].reshape(n, k, s).std(2)
        tracks["hf_floor"] = np.log(np.percentile(sub, 10, axis=1) + 1e-6)
        tracks["hf"] = _logrms(hf) - _logrms(fr)

    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) ** 2 + 1e-9
    f = np.fft.rfftfreq(fr.shape[1], 1 / sr)
    hi = min(10000, sr / 2 - 100)
    m = (f > 4000) & (f < hi)
    if m.sum() > 4:
        tracks["flat_hf"] = np.mean(np.log(spec[:, m]), 1) - np.log(np.mean(spec[:, m], 1))
    for hz in (50, 60):
        on = (np.abs(f - hz) < 1.5) | (np.abs(f - 2 * hz) < 1.5) | (np.abs(f - 3 * hz) < 1.5)
        off = ((np.abs(f - hz) > 3) & (np.abs(f - hz) < 10)) | ((np.abs(f - 2 * hz) > 3) & (np.abs(f - 2 * hz) < 10))
        tracks[f"hum{hz}"] = np.log(spec[:, on].mean(1)) - np.log(spec[:, off].mean(1))

    m = (f > 8000) & (f < hi)
    m2 = (f > 4000) & (f < 8000)
    if m.sum() > 4:
        tracks["hf_tilt"] = np.log(spec[:, m].mean(1)) - np.log(spec[:, m2].mean(1))

    # spectral flux on 23 ms sub-frames: reverb smears note changes
    w2 = 512 if sr <= 24000 else 1024
    k2 = fr.shape[1] // w2
    sf = np.abs(np.fft.rfft(fr[:, : k2 * w2].reshape(n, k2, w2) * np.hanning(w2), axis=2))
    lsf = np.log(sf[:, :, 2: w2 // 4] + 1e-3)
    tracks["flux"] = np.abs(np.diff(lsf, axis=1)).mean((1, 2))

    edges = _mel_edges(60, hi, 26)
    total = np.log(spec[:, (f > 40) & (f < hi)].sum(1))
    for b in range(24):
        lo_, c_, hi_ = edges[b], edges[b + 1], edges[b + 2]
        wts = np.clip(np.minimum((f - lo_) / (c_ - lo_), (hi_ - f) / (hi_ - c_)), 0, None)
        tracks[f"mel{b:02d}"] = np.log((spec * wts).sum(1) + 1e-9) - total
    return tracks, np.arange(n) * hop + win / 2


def _standardise(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    sd = mad if mad > 1e-9 else (v.std() + 1e-9)
    return np.clip((v - med) / sd, -6, 6)


def whiten(Z, shrink=0.05):
    # tracks are strongly correlated (neighbouring bands move together with
    # the melody); decorrelating them keeps one musical change from being
    # counted twenty times, so a change in a few channel tracks can win
    Z = Z - Z.mean(0)
    Z = Z / (Z.std(0) + 1e-9)
    C = (1 - shrink) * np.cov(Z.T) + shrink * np.eye(Z.shape[1])
    w, V = np.linalg.eigh(C)
    return Z @ (V @ np.diag(w ** -0.5) @ V.T)


def scan(Z, minlen, maxlen):
    """Z: (n_frames, n_feats), zero-mean. Best segment by summed squared t."""
    n, d = Z.shape
    c = np.vstack([np.zeros(d), np.cumsum(Z, 0)])
    best = (-1.0, 0, n)
    for L in range(minlen, min(maxlen, n - 2) + 1):
        s = c[L:] - c[:-L]
        # overall mean is 0, so mean_in - mean_out = s/L * N/(N-L)
        t = (s / L) * (n / (n - L)) * np.sqrt(L * (n - L) / n)
        score = (t ** 2).sum(1)
        k = int(np.argmax(score))
        if score[k] > best[0]:
            best = (float(score[k]), k, k + L)
    return best


def detect(x, sr):
    tracks, t = features(x, sr)
    Z = np.column_stack([_standardise(tracks[k]) for k in sorted(tracks)])
    Z = Z[:, Z.std(0) > 1e-9]
    _, i, j = scan(whiten(Z), int(MIN_LEN / HOP), int(MAX_LEN / HOP))
    start = t[i] - HOP / 2
    end = t[j - 1] + HOP / 2
    return max(0.0, start), min(len(x) / sr, end)


def load(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if x.dtype.kind == "f":
        x = x * 32768.0
    return sr, x.astype(float)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test-input.wav"
    sr, x = load(path)
    start, end = detect(x, sr)
    with open("detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerow([f"{start:.3f}", f"{end:.3f}"])


if __name__ == "__main__":
    main()
