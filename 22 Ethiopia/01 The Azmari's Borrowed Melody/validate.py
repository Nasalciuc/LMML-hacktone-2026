"""Check main.detect on the training file and on splices made from it.

    python validate.py ../task_ethiopia [n_per_kind] [seed]

Host = the training recording with the known replaced part cut out, then
either rotated or cut into 3-6 s chunks and reshuffled (new musical
structure, same recording chain). Pieces are pasted in RMS-matched with
30 ms crossfades, 1-30 s long, a third of them touching or nearly touching
an edge of the file.

Kinds:
  foreign   piece of the real replaced phrase (<= 6 s, that's all there is)
  resample  host piece through a lower sample rate and back
  codec     host piece low-passed hard at 7.5-10 kHz (lossy-codec style)
  eq        host piece through a different mic EQ (+-3-6 dB peak or shelf)
  pitch     host piece resampled by 0.5-3 % (tiny pitch/speed change)
  clean     host piece high-passed (no drift/DC) + new hiss
  reverb    host piece through a short synthetic room
  hum       host piece with mains hum added
  noisier   host piece with extra hiss
A case passes when both edges are within 1.5 s.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal

import main as det

TOL = 1.5
KINDS = ["foreign", "resample", "codec", "eq", "pitch", "clean", "reverb", "hum", "noisier"]


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


def shuffled(h, rng, sr):
    cuts = [0]
    while cuts[-1] < len(h):
        cuts.append(cuts[-1] + int(rng.uniform(3, 6) * sr))
    chunks = [h[cuts[k]:cuts[k + 1]] for k in range(len(cuts) - 1)]
    rng.shuffle(chunks)
    f = int(0.03 * sr)
    out = chunks[0]
    for c in chunks[1:]:
        if len(c) <= f:
            continue
        out = np.concatenate([out[:-f], out[-f:] * np.linspace(1, 0, f) + c[:f] * np.linspace(0, 1, f), c[f:]])
    return out


def peaking(f0, gain_db, q, sr):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def make_piece(kind, host, foreign, L, rng, sr):
    if kind == "foreign":
        a = rng.integers(0, len(foreign) - L + 1)
        return foreign[a:a + L]
    a = rng.integers(0, len(host) - L - int(0.1 * sr))
    p = host[a:a + L + int(0.1 * sr)].copy()
    lvl = p.std()
    if kind == "resample":
        down = int(rng.choice([16000, 18000, 19000])) if sr <= 24000 else int(rng.choice([32000, 22050, 24000]))
        g = np.gcd(down, sr)
        p = signal.resample_poly(signal.resample_poly(p, down // g, sr // g), sr // g, down // g)
    elif kind == "codec":
        fc = min(rng.uniform(7500, 10000), sr / 2 * 0.95)
        p = signal.sosfiltfilt(signal.butter(10, fc, "lp", fs=sr, output="sos"), p)
    elif kind == "eq":
        if rng.random() < 0.5:
            b, a_ = peaking(rng.uniform(300, 4000), rng.choice([-1, 1]) * rng.uniform(3, 6), 1.0, sr)
            p = signal.lfilter(b, a_, p)
        else:
            sh = signal.butter(1, rng.uniform(500, 3000), "hp" if rng.random() < 0.5 else "lp", fs=sr, output="sos")
            p = p + rng.uniform(-0.5, 0.5) * signal.sosfilt(sh, p)
    elif kind == "pitch":
        r = 1 + rng.choice([-1, 1]) * rng.uniform(0.005, 0.03)
        up, dn = int(round(1000 * r)), 1000
        g = np.gcd(up, dn)
        p = signal.resample_poly(p, up // g, dn // g)
    elif kind == "clean":
        p = signal.sosfiltfilt(signal.butter(4, 30, "hp", fs=sr, output="sos"), p)
        p = p + rng.normal(0, lvl * 10 ** (rng.uniform(-40, -28) / 20), len(p))
    elif kind == "reverb":
        n = int(rng.uniform(0.2, 0.6) * sr)
        ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (rng.uniform(0.05, 0.15) * sr))
        ir[0] = 1.0
        p = signal.fftconvolve(p, ir)[:len(p)]
    elif kind == "hum":
        hz = rng.choice([50.0, 60.0])
        tt = np.arange(len(p)) / sr
        p = p + lvl * 10 ** (rng.uniform(-36, -26) / 20) * (np.sin(2 * np.pi * hz * tt) + 0.5 * np.sin(4 * np.pi * hz * tt + 1))
    elif kind == "noisier":
        p = p + rng.normal(0, lvl * 10 ** (rng.uniform(-34, -26) / 20), len(p))
    return p[:L]


def one_case(args):
    kind, seed, host, foreign, sr = args
    rng = np.random.default_rng(seed)
    h = shuffled(host, rng, sr) if rng.random() < 0.5 else np.roll(host, rng.integers(len(host)))
    hi = min(30.0, len(foreign) / sr) if kind == "foreign" else 30.0
    L = int(rng.uniform(1.0, hi) * sr)
    u = rng.random()
    if u < 1 / 6:
        pos = int(rng.uniform(0, 1.0) * sr)
    elif u < 1 / 3:
        pos = len(h) - L - int(rng.uniform(0, 1.0) * sr)
    else:
        pos = int(rng.integers(0, len(h) - L))
    y = xfade_insert(h, make_piece(kind, h, foreign, L, rng, sr), pos, sr)
    ds, de = det.detect(y, sr)
    es, ee = abs(ds - pos / sr), abs(de - (pos + L) / sr)
    edge = pos < 1.0 * sr or pos + L > len(h) - 1.0 * sr
    # a stretch touching one end and the rest of the file differ by the same
    # amount; when the splice is the longer of the two, the detector is
    # built to return the shorter one, so count that separately
    dur = len(h) / sr
    comp = (pos + L) / sr if pos < 1.0 * sr else 0.0, pos / sr if pos >= 1.0 * sr else dur
    ambiguous = edge and L / sr > dur / 2 - 1.0
    comp_ok = ambiguous and abs(ds - comp[0]) <= TOL and abs(de - comp[1]) <= TOL
    return kind, L / sr, edge, max(es, ee), pos / sr, (pos + L) / sr, ds, de, dur, ambiguous, comp_ok


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "../task_ethiopia")
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sr, x = det.load(str(data / "train-input.wav"))
    s0, e0 = np.loadtxt(data / "train-output.csv", delimiter=",", skiprows=1)

    s, e = det.detect(x, sr)
    print(f"train: detected {s:.2f}-{e:.2f}  true {s0:.2f}-{e0:.2f}  err {abs(s - s0):.2f}/{abs(e - e0):.2f}")

    a, b = int(s0 * sr), int(e0 * sr)
    foreign = x[a:b]
    f = int(0.03 * sr)
    head, tail = x[:a].copy(), x[b:].copy()
    host = np.concatenate([head[:-f], head[-f:] * np.linspace(1, 0, f) + tail[:f] * np.linspace(0, 1, f), tail[f:]])

    jobs = [(k, seed0 * 100000 + i * 100 + KINDS.index(k), host, foreign, sr) for k in KINDS for i in range(n_per)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one_case, jobs))

    ok = lambda r: r[3] <= TOL
    for k in KINDS:
        rk = [r for r in res if r[0] == k]
        errs = [r[3] for r in rk]
        print(f"{k:9s} {sum(map(ok, rk)):3d}/{len(rk)}  median max-err {np.median(errs):.2f}s")
    for name, sel in [("1-3 s", lambda r: r[1] < 3), ("3-10 s", lambda r: 3 <= r[1] < 10),
                      ("10-30 s", lambda r: r[1] >= 10), ("at an edge", lambda r: r[2])]:
        rk = [r for r in res if sel(r)]
        print(f"{name:10s} {sum(map(ok, rk)):3d}/{len(rk)}")
    print(f"all synthetic: {sum(map(ok, res))}/{len(res)}")
    amb = [r for r in res if r[9]]
    print(f"of which ambiguous (splice > half the file, at an edge): {len(amb)}, "
          f"exact {sum(map(ok, amb))}, complement returned {sum(r[10] for r in amb)}")
    clear = [r for r in res if not r[9]]
    print(f"unambiguous cases: {sum(map(ok, clear))}/{len(clear)}")
    if "-v" in sys.argv:
        for r in res:
            if not ok(r):
                print(f"  miss {r[0]:9s} true {r[4]:5.1f}-{r[5]:5.1f} got {r[6]:5.1f}-{r[7]:5.1f} of {r[8]:.0f}s")


if __name__ == "__main__":
    main()
