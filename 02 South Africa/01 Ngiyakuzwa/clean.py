"""Ngiyakuzwa? - clean the stadium voice note.

Noise in voice_note.wav (measured, see README):
  * two vuvuzela-like horns: steady harmonic combs at 217.5 Hz x k and 233.1 Hz x k,
    switching on and off in blocks of several seconds;
  * crowd roar / cheering: broadband, slowly varying;
  * a hiss band around 5-7 kHz.

Pipeline:
  1. STFT (64 ms window, 8 ms hop).
  2. Noise power per frequency bin = running low percentile over +-1.5 s (minimum-statistics
     style): stationary things (horn harmonics, roar floor) end up in it, speech syllables
     (short, moving) do not.
  3. Extra horn suppression: bins on the two harmonic combs are attenuated whenever the horn
     is on in that frame.
  4. Wiener-type gain with over-subtraction and a floor, smoothed over time/frequency.
  5. Band-pass 120-7000 Hz, level normalisation.

Writes cleaned/voice_note_clean.wav (+ a voice-band-only version) and work/noise_profile.npz.
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "task_south_african_republic", "voice_note.wav")
OUT = os.path.join(HERE, "cleaned")

NPERSEG, HOP = 1024, 128
HORN_F0 = (217.5, 233.1)


def stft(x, sr):
    f, t, Z = signal.stft(x, sr, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                          boundary="even", padded=True)
    return f, t, Z


def istft(Z, sr, n):
    _, y = signal.istft(Z, sr, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                        boundary=True)
    return y[:n]


def running_percentile(P, frames, q, step=4):
    """Percentile over a sliding time window, computed on a decimated grid then interpolated."""
    sub = P[:, ::step]
    w = max(3, frames // step) | 1
    est = ndimage.percentile_filter(sub, q, size=(1, w), mode="nearest")
    idx = np.arange(P.shape[1]) / step
    lo = np.clip(np.floor(idx).astype(int), 0, sub.shape[1] - 1)
    hi = np.clip(lo + 1, 0, sub.shape[1] - 1)
    a = idx - np.floor(idx)
    return est[:, lo] * (1 - a) + est[:, hi] * a


def horn_mask(f, P):
    """Per (bin, frame) weight 0..1 of how much a bin belongs to an active horn harmonic."""
    M = np.zeros_like(P, dtype=np.float32)
    df = f[1] - f[0]
    for f0 in HORN_F0:
        k = np.arange(1, int(7600 // f0) + 1)
        fk = k * f0
        # horn activity per frame: energy at the first 4 harmonics vs. neighbours
        on = []
        for h in fk[:4]:
            i = int(round(h / df))
            peak = P[i - 1:i + 2].sum(0)
            side = P[i - 5:i - 2].sum(0) + P[i + 3:i + 6].sum(0)
            on.append(peak / (side + 1e-12))
        active = np.median(np.array(on), axis=0) > 4.0
        active = ndimage.binary_dilation(active, iterations=8)
        for h in fk:
            width = 1 + int(h / 1500)           # harmonics spread a little at high k
            i = int(round(h / df))
            M[max(i - width, 0):i + width + 1, active] = 1.0
    return M


def steady_lines(P, hold_s=0.6, ratio=3.0):
    """Mask of steady tonal lines (any horn / whistle): a bin that is a spectral peak
    (power > ratio x median of its +-4 neighbouring bins) for most of a +-hold_s/2 window.
    Speech harmonics glide and do not stay on one bin that long."""
    Pl = np.log(P + 1e-20)
    med = ndimage.median_filter(Pl, size=(9, 1), mode="nearest")
    peak = (Pl - med > np.log(ratio)).astype(np.float32)
    frames = int(hold_s * 16000 / HOP)
    steady = ndimage.uniform_filter1d(peak, frames, axis=1, mode="nearest") > 0.6
    return ndimage.binary_dilation(steady, structure=np.ones((3, 1), bool))


def clean(x, sr, over=2.0, floor=0.06, q=25, win_s=3.0):
    f, t, Z = stft(x, sr)
    P = np.abs(Z) ** 2
    frames = int(win_s * sr / HOP)
    N = running_percentile(P, frames, q) * 1.6        # percentile -> mean-ish noise power
    Hm = np.maximum(horn_mask(f, P), steady_lines(P))
    N = np.maximum(N, Hm * P * 0.95)                  # tonal bins: treat ~all energy as noise
    snr = np.maximum(P / (N + 1e-12) - 1.0, 0.0)
    G = snr / (snr + over)                            # Wiener-like gain
    G = ndimage.uniform_filter(G, size=(3, 5))        # tame musical noise
    G = np.maximum(G, floor)
    G = G * np.where(Hm > 0, 0.03, 1.0)               # tonal lines: deep cut after smoothing
    G[f < 120] = 0.0
    G[f > 7000] *= 0.2
    y = istft(Z * G, sr, len(x))
    return y, dict(f=f, t=t, N=N[:, ::50].astype(np.float32), G=G, P=P)


def normalise(y, peak=0.9):
    return y / (np.max(np.abs(y)) + 1e-12) * peak


def voice_band(y, sr, lo=250, hi=3800):
    b, a = signal.butter(4, [lo, hi], btype="band", fs=sr)
    return signal.filtfilt(b, a, y)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    x, sr = sf.read(src)
    os.makedirs(OUT, exist_ok=True)
    y, info = clean(x, sr)
    sf.write(os.path.join(OUT, "voice_note_clean.wav"), normalise(y), sr, subtype="PCM_16")
    sf.write(os.path.join(OUT, "voice_note_clean_voiceband.wav"), normalise(voice_band(y, sr)), sr,
             subtype="PCM_16")
    os.makedirs(os.path.join(HERE, "work"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "work", "noise_profile.npz"), f=info["f"], N=info["N"])
    del info
    # harder setting: less noise left, more "underwater" artefacts on the voice
    y2, _ = clean(x, sr, over=4.0, floor=0.02, q=35)
    sf.write(os.path.join(OUT, "voice_note_clean_strong.wav"), normalise(voice_band(y2, sr, 200, 5000)),
             sr, subtype="PCM_16")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
