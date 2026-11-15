import sys, numpy as np, soundfile as sf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

a, b, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
srcs = sys.argv[4:] or ["../task_south_african_republic/voice_note.wav", "cleaned/voice_note_clean.wav"]
fig, ax = plt.subplots(len(srcs), 1, figsize=(18, 4.2 * len(srcs)), sharex=True)
ax = np.atleast_1d(ax)
for i, s in enumerate(srcs):
    x, sr = sf.read(s)
    x = x[int(a * sr):int(b * sr)]
    f, t, Z = stft(x, sr, nperseg=512, noverlap=512 - 64)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    ax[i].pcolormesh(t + a, f, S, vmin=S.max() - 70, vmax=S.max(), cmap="magma", shading="auto")
    ax[i].set_ylim(0, 8000)
    ax[i].set_title(s)
ax[-1].set_xticks(np.arange(np.ceil(a * 10) / 10, b, 0.1), minor=True)
ax[-1].set_xticks(np.arange(np.ceil(a * 2) / 2, b, 0.5))
for x_ in ax:
    x_.grid(True, which="both", axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(out, dpi=70)
