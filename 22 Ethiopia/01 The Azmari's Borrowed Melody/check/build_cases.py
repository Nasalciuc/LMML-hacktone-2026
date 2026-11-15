"""Builds the test recordings and stores the detector's frame features for each.

    python build_cases.py <train-input.wav> <cache folder> [processes]

Writes cases.pkl (about 115 MB) into the cache folder, so keep that folder
outside the repo. About 3 s per recording: roughly 5 minutes with 3 processes.

344 recordings with a known replaced part, plus one control with none:

  first 140   32 made from the train file and 108 synthetic masenqo/washint
              recordings from the archive's generator, replaced part 2.5-10 s
  fresh 204   synthetic only, other seeds: 2.5-10 s, very short (1.5-2.5 s),
              long (10-14 s and 14-20 s), at the very start, at the very end

Every synthetic group has equal thirds: another instrument / same instrument
through another recording chain / same instrument through the same chain.

The features are the same in arhiva/main.py and in ../main.py (compare.py
checks that), so one cache serves both.
"""
import os
import pickle
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import archive, synth as G

SCEN = ["diff_inst", "same_inst_diff_chain", "same_inst_same_chain"]
T0, T1 = 14.96, 20.965  # train-output.csv
FRESH = [  # group, recordings per scenario, length range (s), position, seed base
    ("S3 fresh 2.5-10 s", 24, (2.5, 10.0), None, 900000),
    ("short 1.5-2.5 s", 10, (1.5, 2.5), None, 700000),
    ("long 10-14 s", 10, (10.0, 14.0), None, 600000),
    ("long 14-20 s", 8, (14.0, 20.0), None, 500000),
    ("at file start", 8, (3.0, 8.0), "start", 400000),
    ("at file end", 8, (3.0, 8.0), "end", 300000),
]


def train_cases(train_path):
    """33 recordings from the train file: (group, name, samples, rate, truth)."""
    sr, xi = wavfile.read(train_path)
    x = xi.astype(float)
    a, b = int(T0 * sr), int(T1 * sr)
    L0 = b - a
    host = np.concatenate([x[:a], x[b:]])  # the take without the replaced part
    HD = len(host) / sr
    lo = signal.butter(6, 150, "low", fs=sr, output="sos")
    hi = signal.butter(6, 170, "high", fs=sr, output="sos")
    phrase = x[a:b]
    rum = signal.sosfiltfilt(lo, phrase)   # the 20-150 Hz part of the foreign phrase
    mus = signal.sosfiltfilt(hi, phrase)   # its music without that part

    def fade(L):
        return np.minimum(1, np.minimum(np.arange(L), L - np.arange(L)) / (0.03 * sr))

    def put(y, p, piece):
        L = len(piece)
        f = fade(L)
        y[p:p + L] = y[p:p + L] * (1 - f) + piece * f

    rng = np.random.default_rng(7)
    out = [("A train", "train", x, sr, (T0, T1)),
           ("H host only (no splice)", "host only", host, sr, None)]
    for p_s in (0.6, 3.0, 9.0, 24.0, 33.0, 41.0, HD - L0 / sr - 0.6):
        y = host.copy()
        p = int(p_s * sr)
        put(y, p, phrase)
        out.append(("B phrase moved", f"phrase at {p_s:.1f}", y, sr, (p / sr, (p + L0) / sr)))
    for _ in range(5):
        L = int(rng.uniform(3.0, 7.0) * sr)
        p = int(rng.uniform(2, HD - 9) * sr)
        for g in (0, -10, -16):
            y = host.copy()
            y[p:p + L] += np.resize(rum, L) * 10 ** (g / 20) * fade(L)
            out.append(("C same notes + rumble", f"rumble {g:+d} dB at {p / sr:.1f}", y, sr, (p / sr, (p + L) / sr)))
    for p_s in (5.0, 22.0, 38.0):
        y = host.copy()
        p = int(p_s * sr)
        put(y, p, mus + signal.sosfiltfilt(lo, host[p:p + L0]))
        out.append(("D phrase without rumble", f"music-only at {p_s:.0f}", y, sr, (p / sr, (p + L0) / sr)))
    for p_s, q_s in ((5.0, 30.0), (22.0, 44.0), (38.0, 8.0)):
        y = host.copy()
        p = int(p_s * sr)
        put(y, p, phrase)
        q = int(q_s * sr)
        Ld = int(4.0 * sr)
        put(y, q, mus[:Ld] + signal.sosfiltfilt(lo, y[q:q + Ld]))
        out.append(("E phrase + music decoy", f"phrase {p_s:.0f}, decoy {q_s:.0f}", y, sr, (p / sr, (p + L0) / sr)))
    for p_s, q_s in ((6.0, 30.0), (25.0, 44.0), (40.0, 10.0)):
        y = host.copy()
        Lr = int(5.0 * sr)
        p = int(p_s * sr)
        y[p:p + Lr] += np.resize(rum, Lr) * 10 ** (-10 / 20) * fade(Lr)
        q = int(q_s * sr)
        Ld = int(4.0 * sr)
        up = signal.resample(host[q:q + 2 * Ld], Ld)  # host played an octave up
        put(y, q, signal.sosfiltfilt(hi, up) + signal.sosfiltfilt(lo, host[q:q + Ld]))
        out.append(("F weak rumble + octave decoy", f"splice {p_s:.0f}, decoy {q_s:.0f}", y, sr, (p / sr, (p + Lr) / sr)))
    return out


def fresh_case(seed, scenario, Lr, edge):
    """Like the generator's make_case, with a chosen length range and position."""
    rng = np.random.default_rng(seed)
    total = rng.uniform(45, 62)
    L = rng.uniform(*Lr)
    instA = rng.choice(["masenqo", "washint"])
    instB = ("washint" if instA == "masenqo" else "masenqo") if scenario == "diff_inst" else instA
    cA = G.random_chain(rng)
    cB = cA if scenario == "same_inst_same_chain" else G.random_chain(rng)
    tA = G.make_take(rng, instA, total)
    tB = G.make_take(rng, instB, L + 2)
    A = G.chain(rng, tA, cA)
    B = G.chain(rng, tB, cB)
    st = rng.uniform(0, 1.5)
    ins = B[int(st * G.SR):int((st + L) * G.SR)]
    if edge == "start":
        pos = int(rng.uniform(0.0, 0.4) * G.SR)
    elif edge == "end":
        pos = len(A) - len(ins) - int(rng.uniform(0.0, 0.4) * G.SR)
    else:
        pos = int(rng.uniform(0.5, total - L - 0.5) * G.SR)
    y = G.splice(A, ins, pos, rng.choice([0.01, 0.03, 0.06, 0.12]))
    y = y / np.max(np.abs(y)) * rng.uniform(1500, 12000)
    return np.clip(np.round(y), -32768, 32767).astype(float), pos / G.SR, (pos + len(ins)) / G.SR


def specs(train_path):
    out = [("train", train_path, i) for i in range(33)]
    for tag, n, base in (("S1", 12, 1000), ("S2", 24, 50000)):
        for sc in SCEN:
            for s in range(n):
                out.append(("synth", tag, sc, base + 37 * s + len(sc), s))
    for tag, n, Lr, edge, base in FRESH:
        for sc in SCEN:
            for s in range(n):
                out.append(("fresh", tag, sc, base + 101 * s + len(sc), Lr, edge))
    return out


_TRAIN = {}


def build(spec):
    if spec[0] == "train":
        if spec[1] not in _TRAIN:
            _TRAIN[spec[1]] = train_cases(spec[1])
        group, name, y, sr, truth = _TRAIN[spec[1]][spec[2]]
        st, sc = "first 140", "train-made"
    elif spec[0] == "synth":
        _, tag, sc, seed, s = spec
        yi, a, b, m = G.make_case(seed, sc)
        y, sr, truth, st = yi.astype(float), G.SR, (a, b), "first 140"
        group, name = f"{tag} {sc}", f"#{s:02d} {m['instA'][:4]}->{m['instB'][:4]}"
    else:
        _, tag, sc, seed, Lr, edge = spec
        y, a, b = fresh_case(seed, sc, Lr, edge)
        sr, truth, st, group, name = G.SR, (a, b), "fresh 204", tag, f"seed {seed}"
    dur = len(y) / sr
    tt, F = archive.compute_features(y, sr, hop_s=max(0.05, dur / 1500.0))  # as in detect()
    return dict(set=st, group=group, sc=sc, name=name, truth=truth, dur=dur, sr=sr, tt=tt, F=F)


if __name__ == "__main__":
    from multiprocessing import Pool

    train, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out, exist_ok=True)
    todo = specs(train)
    cases = []
    with Pool(nproc) as pool:
        for i, c in enumerate(pool.imap(build, todo, chunksize=1)):
            cases.append(c)
            if i % 25 == 0 or i == len(todo) - 1:
                print(f"{i + 1}/{len(todo)}  {c['set']} / {c['group']}", flush=True)
    with open(os.path.join(out, "cases.pkl"), "wb") as fh:
        pickle.dump(cases, fh, protocol=4)
    print("recordings:", len(cases), "| with a known replaced part:", sum(c["truth"] is not None for c in cases))
