import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
sos = signal.butter(4, [10, 150], btype='band', fs=sr, output='sos')
lf = signal.sosfiltfilt(sos, x)
sos2 = signal.butter(4, 8000, btype='high', fs=sr, output='sos')
hf = signal.sosfiltfilt(sos2, x)
fig, ax = plt.subplots(4, 1, figsize=(16, 14))
for k, (t0, t1) in enumerate([(14.0, 18.0), (18.0, 22.0)]):
    sl = slice(int(t0*sr), int(t1*sr))
    tt = np.arange(sl.start, sl.stop)/sr
    ax[2*k].plot(tt, x[sl], lw=0.3, color='gray')
    ax2 = ax[2*k].twinx(); ax2.plot(tt, lf[sl], color='r', lw=0.8)
    ax[2*k+1].plot(tt, hf[sl], lw=0.3)
    for a_ in (ax[2*k], ax[2*k+1]):
        for v in (14.96, 20.965): 
            if t0 <= v <= t1: a_.axvline(v, color='k')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
