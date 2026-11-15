import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
seg = x[int(t0*sr):int(t1*sr)]
fig, ax = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
f, t, S = signal.spectrogram(seg, sr, nperseg=2048, noverlap=2048-256)
t = t + t0
m = f < 1500
ax[0].pcolormesh(t, f[m], 10*np.log10(S[m]+1e-12), shading='auto', vmin=-30, vmax=50)
m = f < 200
ax[1].pcolormesh(t, f[m], 10*np.log10(S[m]+1e-12), shading='auto', vmin=-40, vmax=20)
b, a = signal.butter(2, 5/(sr/2))
lp = signal.filtfilt(b, a, x)[int(t0*sr):int(t1*sr)]
tt = np.arange(len(seg))/sr + t0
ax[2].plot(tt, lp)
env = np.sqrt(signal.filtfilt(*signal.butter(2, 20/(sr/2)), seg**2).clip(0))
ax[3].plot(tt, env)
for a_ in ax:
    for v in (14.96, 20.965): a_.axvline(v, color='r')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
