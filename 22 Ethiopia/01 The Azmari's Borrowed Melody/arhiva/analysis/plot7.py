import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
tr = np.load(sys.argv[3])
t, f0, c, r = tr.T
midi = 69 + 12*np.log2(f0/440)
sos = signal.butter(2, 8, fs=sr, output='sos')
lp = signal.sosfiltfilt(sos, x)
env = np.sqrt(np.maximum(signal.sosfiltfilt(signal.butter(2, 8, fs=sr, output='sos'), x**2), 0))
tt = np.arange(len(x))/sr
fig, ax = plt.subplots(4, 1, figsize=(18, 14))
for k, (a, b) in enumerate([(0, 14), (14, 28), (28, 42), (42, 56)]):
    m = (tt >= a) & (tt < b)
    ax[k].plot(tt[m][::50], lp[m][::50], color='b', lw=0.8)
    ax[k].plot(tt[m][::50], env[m][::50]/100, color='gray', lw=0.8)
    ax2 = ax[k].twinx()
    mm = (t >= a) & (t < b) & (c > 0.85)
    ax2.plot(t[mm], midi[mm], '.', color='r', ms=2)
    ax2.set_ylim(55, 80)
    for v in (14.96, 20.965):
        if a <= v <= b: ax[k].axvline(v, color='k')
    ax[k].set_xlim(a, b)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=65)
