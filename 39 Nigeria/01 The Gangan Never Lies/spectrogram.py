import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

path = sys.argv[1] if len(sys.argv) > 1 else "../task_nigeria/challenge.wav"
sr, x = wavfile.read(path)
f, t, S = spectrogram(x.astype(float), fs=sr, nperseg=512, noverlap=448)
band = (f > 2900) & (f < 3800)  # the only band with extra energy

plt.figure(figsize=(16, 4))
plt.pcolormesh(t, f[band], 10 * np.log10(S[band] + 1e-9), cmap="gray")
plt.xlabel("s")
plt.ylabel("Hz")
plt.tight_layout()
plt.savefig("spectrogram.png", dpi=80)
