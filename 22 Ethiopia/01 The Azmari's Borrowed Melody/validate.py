"""Check main.detect on the training file and on splices made from it.

    python validate.py ../task_ethiopia [n_per_kind]

Host = the training recording with the known replaced part cut out.
Kinds of synthetic splice (all RMS-matched, 30 ms crossfades, 2.5-8 s):
  foreign   a piece of the real replaced part pasted elsewhere
  clean     another piece of the host, high-passed (no drift/DC) + new hiss
  reverb    another piece of the host through a short synthetic room
  hum       another piece of the host with mains hum added
  dull      another piece of the host low-passed at 6-7 kHz + new hiss
  noisier   another piece of the host with extra hiss only
A case passes when both edges are within 1.5 s.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

import main as det

TOL = 1.5


def xfade_insert(host, piece, pos, sr, fade=0.03):
    y = host.copy()
    L = len(piece)
    seg = host[pos:pos + L]
    piece = piece * (seg.std() / (piece.std() + 1e-9))
    f = int(fade * sr)
    ramp = np.ones(L)
    ramp[:f] = np.linspace(0, 1, f)
    ramp[-f:] = np.linspace(1, 0, f)
    y[pos:pos + L] = seg * (1 - ramp) + piece * ramp
    return y


def make_piece(kind, host, foreign, L, rng, sr):
    if kind == "foreign":
        a = rng.integers(0, len(foreign) - L + 1)
        return foreign[a:a + L]
    a = rng.integers(0, len(host) - L)
    p = host[a:a + L].copy()
    lvl = p.std()
    if kind == "clean":
        p = signal.sosfiltfilt(signal.butter(4, 30, "hp", fs=sr, output="sos"), p)
        p = p + rng.normal(0, lvl * 10 ** (rng.uniform(-40, -28) / 20), L)
    elif kind == "reverb":
        n = int(rng.uniform(0.2, 0.6) * sr)
        ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (rng.uniform(0.05, 0.15) * sr))
        ir[0] = 1.0
        p = signal.fftconvolve(p, ir)[:L]
    elif kind == "hum":
        hz = rng.choice([50.0, 60.0])
        tt = np.arange(L) / sr
        p = p + lvl * 10 ** (rng.uniform(-36, -26) / 20) * (np.sin(2 * np.pi * hz * tt) + 0.5 * np.sin(4 * np.pi * hz * tt + 1))
    elif kind == "dull":
        p = signal.sosfiltfilt(signal.butter(6, rng.uniform(6000, 7000), "lp", fs=sr, output="sos"), p)
        p = p + rng.normal(0, lvl * 10 ** (-45 / 20), L)
    elif kind == "noisier":
        p = p + rng.normal(0, lvl * 10 ** (rng.uniform(-34, -26) / 20), L)
    return p


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "../task_ethiopia")
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    sr, x = det.load(str(data / "train-input.wav"))
    s0, e0 = np.loadtxt(data / "train-output.csv", delimiter=",", skiprows=1)

    s, e = det.detect(x, sr)
    print(f"train: detected {s:.2f}-{e:.2f}  true {s0:.2f}-{e0:.2f}  err {abs(s - s0):.2f}/{abs(e - e0):.2f}")

    a, b = int(s0 * sr), int(e0 * sr)
    foreign = x[a:b]
    f = int(0.03 * sr)
    tail = x[b:].copy()
    tail[:f] *= np.linspace(0, 1, f)
    head = x[:a].copy()
    head[-f:] *= np.linspace(1, 0, f)
    host = np.concatenate([head[:-f], head[-f:] + tail[:f], tail[f:]])

    rng = np.random.default_rng(1)
    total_ok = total = 0
    for kind in ["foreign", "clean", "reverb", "hum", "dull", "noisier"]:
        ok, errs = 0, []
        for _ in range(n_per):
            shift = rng.integers(0, len(host))
            h = np.roll(host, shift)
            L = int(rng.uniform(2.5, min(8.0, len(foreign) / sr if kind == "foreign" else 8.0)) * sr)
            pos = int(rng.integers(int(2 * sr), len(h) - L - int(2 * sr)))
            y = xfade_insert(h, make_piece(kind, h, foreign, L, rng, sr), pos, sr)
            ds, de = det.detect(y, sr)
            es, ee = abs(ds - pos / sr), abs(de - (pos + L) / sr)
            errs.append(max(es, ee))
            ok += es <= TOL and ee <= TOL
        total_ok += ok
        total += n_per
        print(f"{kind:8s} {ok}/{n_per}  median max-err {np.median(errs):.2f}s  worst {max(errs):.2f}s")
    print(f"all synthetic: {total_ok}/{total}")


if __name__ == "__main__":
    main()
