# How deterministic are the horns? Fit sum of harmonics k*f0 (fixed f0, constant complex amp)
# over a block and see how much of the energy it explains; scan f0 finely.
import numpy as np, soundfile as sf
x, sr = sf.read("../task_south_african_republic/voice_note.wav")
a, b = 306.0, 309.8
seg = x[int(a * sr):int(b * sr)]
t = np.arange(len(seg)) / sr


def fit(f0s, seg, t, K=34):
    cols = []
    for f0 in f0s:
        for k in range(1, K + 1):
            f = f0 * k
            if f > 7900: break
            cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    A = np.stack(cols, 1)
    c, *_ = np.linalg.lstsq(A, seg, rcond=None)
    r = seg - A @ c
    return 10 * np.log10(np.sum(seg ** 2) / np.sum(r ** 2)), r


for f0 in np.arange(217.40, 217.61, 0.02):
    g, _ = fit([f0], seg, t)
    print(f"{f0:.2f} {g:.2f}")
for f0 in np.arange(233.00, 233.21, 0.02):
    g, _ = fit([f0], seg, t)
    print(f"{f0:.2f} {g:.2f}")
