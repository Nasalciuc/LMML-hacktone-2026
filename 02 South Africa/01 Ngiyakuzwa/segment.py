"""Find the spoken phrases in the cleaned voice note and export them.

Speech activity = energy of the cleaned signal in the voice band (250-3500 Hz) clearly above
its own running floor. Runs closer than WORD_GAP are joined into words, words closer than
PHRASE_GAP into phrases. Each phrase is written as segments/NNN_<start>-<end>.wav (cleaned)
with a spectrogram PNG (cleaned on top, original underneath); segments/index.csv lists them.

Also scores every word-sized chunk for the shape of "ikhodi" (i-KHO-di) and exports the
~10 s around the best ones to ikhodi_candidates/.
"""
import csv
import os
import shutil

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "..", "task_south_african_republic", "voice_note.wav")
CLEAN = os.path.join(HERE, "cleaned", "voice_note_clean.wav")
SEG_DIR = os.path.join(HERE, "segments")
CAND_DIR = os.path.join(HERE, "ikhodi_candidates")

HOP_S = 0.01
WORD_GAP, PHRASE_GAP = 0.25, 0.7
MIN_PHRASE, PAD = 0.4, 0.25


def frames_db(x, sr, lo, hi):
    b, a = signal.butter(4, [lo, hi], btype="band", fs=sr)
    y = signal.filtfilt(b, a, x)
    hop = int(HOP_S * sr)
    n = len(y) // hop
    e = (y[:n * hop].reshape(n, hop) ** 2).mean(1)
    return 10 * np.log10(e + 1e-12)


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge(rs, gap):
    out = []
    for s, e in rs:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def activity(y, sr):
    v = frames_db(y, sr, 250, 3500)
    v = ndimage.uniform_filter1d(v, 5)
    floor = ndimage.percentile_filter(v, 20, size=int(6 / HOP_S), mode="nearest")
    act = v > floor + 6
    act = ndimage.binary_opening(act, iterations=3)          # drop clicks < 60 ms
    return act, v, floor


def ts(sec):
    return f"{int(sec // 60):02d}m{sec % 60:05.2f}s"


def spectro_png(path, clips, sr, t0, title):
    fig, ax = plt.subplots(len(clips), 1, figsize=(12, 3.2 * len(clips)), squeeze=False)
    for k, (lab, c) in enumerate(clips):
        f, t, S = signal.stft(c, sr, nperseg=512, noverlap=448)
        L = 20 * np.log10(np.abs(S) + 1e-9)
        a = ax[k, 0]
        a.pcolormesh(t + t0, f, L, vmin=L.max() - 65, vmax=L.max(), shading="auto", cmap="magma")
        a.set_ylim(0, 8000)
        a.set_ylabel(f"{lab}\nHz")
    ax[0, 0].set_title(title)
    ax[-1, 0].set_xlabel("time in voice note (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ------------------------------------------------------------- ikhodi shape score
def ikhodi_score(y, sr, s, e):
    """Heuristic score of how much the chunk y[s:e] looks like i-KHO-di:
    vowel with high F2 (i) -> short gap -> aspirated burst (broadband noise, kh) ->
    back vowel with low F2 (o) -> short closure (d) -> vowel with high F2 (i).
    Uses band energies per 10 ms frame; returns (score, time of the burst)."""
    hop = int(HOP_S * sr)
    seg = y[s:e]
    if len(seg) < int(0.3 * sr):
        return 0.0, None
    f, t, S = signal.stft(seg, sr, nperseg=400, noverlap=400 - hop)
    P = np.abs(S) ** 2 + 1e-12
    band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum(0)  # noqa: E731
    low = band(250, 1000)            # F1 / back-vowel energy
    f2hi = band(1800, 3200)          # front vowel (i) F2/F3
    f2lo = band(700, 1300)           # back vowel (o) F2
    asp = band(2500, 6500)           # aspiration / frication
    tot = band(250, 6500)
    db = 10 * np.log10(tot)
    voiced = low / tot > 0.45
    fric = asp / tot > 0.5
    front = f2hi / (f2lo + 1e-12)
    n = len(db)
    best, best_t = 0.0, None
    for b in range(3, n - 25):
        if not fric[b]:
            continue
        # i before the burst (within 150 ms), front quality
        pre = slice(max(0, b - 15), b - 2)
        pre_v = voiced[pre]
        if pre_v.sum() < 3:
            continue
        pre_front = np.median(front[pre][pre_v])
        # dip (closure) right before the burst
        dip = db[max(0, b - 6):b].min() < db[b] - 4
        # aspiration lasts 30-120 ms
        k = b
        while k < n and fric[k]:
            k += 1
        alen = k - b
        if not 4 <= alen <= 12:
            continue
        # o: voiced, back (low F2) within 40-220 ms after the burst
        o = slice(k, min(n, k + 18))
        o_v = voiced[o]
        if o_v.sum() < 5:
            continue
        o_front = np.median(front[o][o_v])
        # d closure + final i (front) in the next 300 ms
        tail = slice(min(n, k + 10), min(n, k + 35))
        t_v = voiced[tail]
        if t_v.sum() < 3:
            continue
        i_front = np.max(ndimage.uniform_filter1d(front[tail], 4))
        sc = (np.log(pre_front + 1e-3) - np.log(o_front + 1e-3)) \
            + (np.log(i_front + 1e-3) - np.log(o_front + 1e-3)) \
            + 0.5 * dip + 0.1 * min(alen, 8)
        if sc > best:
            best, best_t = sc, (s + b * hop) / sr
    return best, best_t


def horn_switch_times(x, sr):
    """Times where the background horns switch on/off (big step in the original's
    200-1000 Hz level): the switch makes a broadband click that can mimic a 'kh' burst."""
    v = ndimage.uniform_filter1d(frames_db(x, sr, 200, 1000), 5)
    step = v[10:] - v[:-10]
    idx = signal.find_peaks(np.abs(step), height=8, distance=50)[0] + 5
    return idx * HOP_S


def main():
    y, sr = sf.read(CLEAN)
    x, _ = sf.read(ORIG)
    switches = horn_switch_times(x, sr)
    act, v, floor = activity(y, sr)
    words = merge(runs(act), int(WORD_GAP / HOP_S))
    phrases = [p for p in merge(words, int(PHRASE_GAP / HOP_S)) if (p[1] - p[0]) * HOP_S >= MIN_PHRASE]

    for d in (SEG_DIR, CAND_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    rows = []
    for k, (s, e) in enumerate(phrases, 1):
        t0 = max(0.0, s * HOP_S - PAD)
        t1 = min(len(y) / sr, e * HOP_S + PAD)
        a, b = int(t0 * sr), int(t1 * sr)
        name = f"{k:03d}_{ts(t0)}-{ts(t1)}"
        clip = y[a:b] / (np.max(np.abs(y[a:b])) + 1e-9) * 0.9
        sf.write(os.path.join(SEG_DIR, name + ".wav"), clip, sr, subtype="PCM_16")
        spectro_png(os.path.join(SEG_DIR, name + ".png"),
                    [("cleaned", y[a:b]), ("original", x[a:b])], sr, t0, name)
        rows.append(dict(n=k, file=name + ".wav", start_s=round(t0, 2), end_s=round(t1, 2),
                         dur_s=round(t1 - t0, 2)))

    # ikhodi candidates: score each word-sized chunk (with 0.1 s margin)
    hop = int(HOP_S * sr)
    scored = []
    for s, e in words:
        sc, tb = ikhodi_score(y, sr, max(0, (s - 10) * hop), min(len(y), (e + 10) * hop))
        if tb is not None and np.min(np.abs(switches - tb)) > 0.2:
            scored.append((sc, tb, s * HOP_S, e * HOP_S))
    scored.sort(reverse=True)
    picked = []
    for sc, tb, ws, we in scored:
        if all(abs(tb - p[1]) > 8 for p in picked):
            picked.append((sc, tb, ws, we))
        if len(picked) == 10:
            break
    cand_rows = []
    for r, (sc, tb, ws, we) in enumerate(picked, 1):
        t0, t1 = max(0.0, tb - 3.0), min(len(y) / sr, tb + 7.0)   # word + what follows it
        a, b = int(t0 * sr), int(t1 * sr)
        name = f"rank{r:02d}_burst_at_{ts(tb)}"
        clip = y[a:b] / (np.max(np.abs(y[a:b])) + 1e-9) * 0.9
        sf.write(os.path.join(CAND_DIR, name + ".wav"), clip, sr, subtype="PCM_16")
        spectro_png(os.path.join(CAND_DIR, name + ".png"),
                    [("cleaned", y[a:b]), ("original", x[a:b])], sr, t0,
                    f"{name}  (score {sc:.2f}; 'kh' burst at {tb:.2f}s)")
        cand_rows.append(dict(rank=r, file=name + ".wav", burst_s=round(tb, 2), clip_start_s=round(t0, 2),
                              clip_end_s=round(t1, 2), score=round(sc, 3)))

    with open(os.path.join(SEG_DIR, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(CAND_DIR, "candidates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0]))
        w.writeheader()
        w.writerows(cand_rows)

    speech = act.sum() * HOP_S
    print(f"{len(switches)} horn on/off switches excluded")
    print(f"speech-active {speech:.0f}s of {len(y) / sr:.0f}s; {len(words)} word chunks; "
          f"{len(phrases)} phrases; top candidates:")
    for c in cand_rows[:10]:
        print(" ", c)


if __name__ == "__main__":
    main()
