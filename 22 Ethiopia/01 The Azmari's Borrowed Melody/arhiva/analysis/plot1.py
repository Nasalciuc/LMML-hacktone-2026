import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
T0, T1 = 14.960, 20.965
fig, ax = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
f, t, S = signal.spectrogram(x, sr, nperseg=4096, noverlap=3072)
m = f < 1200
ax[0].pcolormesh(t, f[m], 10*np.log10(S[m]+1e-12), shading='auto', vmin=-30, vmax=50)
ax[0].set_ylabel('Hz (0-1200)')
m = f < 200
ax[1].pcolormesh(t, f[m], 10*np.log10(S[m]+1e-12), shading='auto', vmin=-40, vmax=20)
ax[1].set_ylabel('Hz (0-200)')
f2, t2, S2 = signal.spectrogram(x, sr, nperseg=1024, noverlap=512)
ax[2].pcolormesh(t2, f2, 10*np.log10(S2+1e-12), shading='auto', vmin=-60, vmax=20)
ax[2].set_ylabel('Hz full')
# low-pass DC drift
b, a = signal.butter(2, 5/(sr/2))
lp = signal.filtfilt(b, a, x)
ax[3].plot(np.arange(len(x))/sr, lp); ax[3].set_ylabel('LP <5Hz')
ax[4].plot(np.arange(len(x))/sr, x, lw=0.2); ax[4].set_ylabel('wave')
for a_ in ax:
    a_.axvline(T0, color='r'); a_.axvline(T1, color='r')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
