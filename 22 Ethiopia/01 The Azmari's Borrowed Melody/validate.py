"""Check main.detect on the training file and on splices made from it.

    python validate.py ../task_ethiopia [n_per_kind] [seed] [-v]

Host = the training recording with the known replaced part cut out, then
either rotated or cut into 3-6 s chunks and reshuffled (new musical
structure, same recording chain). Pieces are pasted in RMS-matched with
30 ms crossfades. Reports the first pass, the second pass and the answer.

realistic: built to look like train. 3-9 s, 20 % at the start, 20 % at the end.
  foreign   the real replaced phrase (whole 6 s or a 3-6 s piece)
  low+18    host piece without sub-20 Hz drift/DC and with a louder 20-150 Hz
  low+9     floor (+12-25, +6-12, +3-6 dB), sometimes a little extra hiss
  low+4
tough: 1-30 s, a third at an edge.
  foreign   piece of the real replaced phrase
  resample  host piece through a lower sample rate and back
  codec     host piece low-passed hard at 7.5-10 kHz
  eq        host piece through a different mic EQ (+-3-6 dB peak or shelf)
  pitch     host piece resampled by 0.5-3 %
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


def low_chain(piece, band_pow, rng, sr, gain_db):
    # what train's real splice looks like: no sub-20 Hz drift/DC, but a
    # louder floor in 20-150 Hz, sometimes a bit of extra hiss
    p = signal.sosfiltfilt(signal.butter(2, 18, "hp", fs=sr, output="sos"), piece)
    nz = signal.sosfiltfilt(signal.butter(2, [20, 150], "bp", fs=sr, output="sos"), rng.normal(0, 1, len(p)))
    p = p + nz * np.sqrt(band_pow * 10 ** (gain_db / 10)) / (nz.std() + 1e-9)
    if rng.random() < 0.5:
        p = p + rng.normal(0, p.std() * 10 ** (rng.uniform(-45, -35) / 20), len(p))
    return p


def realistic_case(kind, h, foreign, rng, sr):
    if kind == "foreign":
        L = len(foreign) if rng.random() < 0.5 else int(rng.uniform(3, 6) * sr)
        a = rng.integers(0, len(foreign) - L + 1)
        piece = foreign[a:a + L]
    else:
        L = int(rng.uniform(3, 9) * sr)
        a = rng.integers(0, len(h) - L)
        band = signal.sosfiltfilt(signal.butter(2, [20, 150], "bp", fs=sr, output="sos"), h).var()
        gain = {"low+18": rng.uniform(12, 25), "low+9": rng.uniform(6, 12), "low+4": rng.uniform(3, 6)}[kind]
        piece = low_chain(h[a:a + L].copy(), band, rng, sr, gain)
    u = rng.random()
    if u < 0.2:
        pos = int(rng.uniform(0, 1.0) * sr)
    elif u < 0.4:
        pos = len(h) - L - int(rng.uniform(0, 1.0) * sr)
    else:
        pos = int(rng.integers(int(1.0 * sr), len(h) - L - int(1.0 * sr)))
    return pos, L, piece


def tough_case(kind, h, foreign, rng, sr):
    hi = min(30.0, len(foreign) / sr) if kind == "foreign" else 30.0
    L = int(rng.uniform(1.0, hi) * sr)
    u = rng.random()
    if u < 1 / 6:
        pos = int(rng.uniform(0, 1.0) * sr)
    elif u < 1 / 3:
        pos = len(h) - L - int(rng.uniform(0, 1.0) * sr)
    else:
        pos = int(rng.integers(0, len(h) - L))
    return pos, L, make_piece(kind, h, foreign, L, rng, sr)


def one_case(args):
    setname, kind, seed, host, foreign, sr = args
    rng = np.random.default_rng(seed)
    h = shuffled(host, rng, sr) if rng.random() < 0.5 else np.roll(host, rng.integers(len(host)))
    pos, L, piece = (realistic_case if setname == "realistic" else tough_case)(kind, h, foreign, rng, sr)
    y = xfade_insert(h, piece, pos, sr)
    pick, (s1, e1, ratio), (s2, e2) = det.detect(y, sr, verbose=True)
    ts, te, dur = pos / sr, (pos + L) / sr, len(h) / sr
    hit = lambda s, e: abs(s - ts) <= TOL and abs(e - te) <= TOL
    # a stretch touching one end and the rest of the file differ by the same
    # amount; when the splice is the longer of the two it can't be told apart
    edge = pos < 1.0 * sr or pos + L > len(h) - 1.0 * sr
    ambiguous = edge and L / sr > dur / 2 - 1.0
    return dict(set=setname, kind=kind, len=L / sr, edge=edge, ambiguous=ambiguous, ratio=ratio,
                first=hit(s1, e1), second=hit(s2, e2), combined=hit(*pick),
                true=(ts, te), got=pick, dur=dur)


REALISTIC = ["foreign", "low+18", "low+9", "low+4"]


def table(res, kinds, title):
    print(f"{title:12s} first  second  combined")
    for k in kinds + ["all"]:
        rk = [r for r in res if k == "all" or r["kind"] == k]
        print(f"  {k:10s} " + "  ".join(f"{sum(r[c] for r in rk):3d}/{len(rk):<3d}" for c in ("first", "second", "combined")))


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "../task_ethiopia")
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sr, x = det.load(str(data / "train-input.wav"))
    s0, e0 = np.loadtxt(data / "train-output.csv", delimiter=",", skiprows=1)

    (s, e), (s1, e1, ratio), (s2, e2) = det.detect(x, sr, verbose=True)
    print(f"train: true {s0:.2f}-{e0:.2f}  first {s1:.2f}-{e1:.2f} (ratio {ratio:.2f})  "
          f"second {s2:.2f}-{e2:.2f}  answer {s:.2f}-{e:.2f}")

    a, b = int(s0 * sr), int(e0 * sr)
    foreign = x[a:b]
    f = int(0.03 * sr)
    head, tail = x[:a].copy(), x[b:].copy()
    host = np.concatenate([head[:-f], head[-f:] * np.linspace(1, 0, f) + tail[:f] * np.linspace(0, 1, f), tail[f:]])

    jobs = [("realistic", k, seed0 * 100000 + 50000 + i * 100 + q, host, foreign, sr)
            for q, k in enumerate(REALISTIC) for i in range(n_per)]
    jobs += [("tough", k, seed0 * 100000 + i * 100 + q, host, foreign, sr)
             for q, k in enumerate(KINDS) for i in range(n_per)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one_case, jobs))

    real = [r for r in res if r["set"] == "realistic"]
    tough = [r for r in res if r["set"] == "tough"]
    table(real, REALISTIC, "realistic")
    print(f"  first-pass ratio on realistic cases: min {min(r['ratio'] for r in real):.2f}, "
          f"median {np.median([r['ratio'] for r in real]):.2f} (answer switches below {det.CONFIDENT})")
    table(tough, KINDS, "tough")
    amb = [r for r in tough if r["ambiguous"]]
    clear = [r for r in tough if not r["ambiguous"]]
    print(f"  tough without the {len(amb)} ambiguous ones (splice > half the file, at an edge): " +
          "  ".join(f"{c} {sum(r[c] for r in clear)}/{len(clear)}" for c in ("first", "second", "combined")))
    if "-v" in sys.argv:
        for r in res:
            if not r["combined"]:
                print(f"  miss {r['set']:9s} {r['kind']:9s} true {r['true'][0]:5.1f}-{r['true'][1]:5.1f} "
                      f"got {r['got'][0]:5.1f}-{r['got'][1]:5.1f} of {r['dur']:.0f}s ratio {r['ratio']:.2f}")


if __name__ == "__main__":
    main()
