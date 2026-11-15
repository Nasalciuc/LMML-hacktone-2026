import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal

def f0_track(x, sr, hop=0.02, win=0.06, fmin=150, fmax=1200):
    H = int(hop*sr); W = int(win*sr)
    out = []
    for i in range(0, len(x)-W, H):
        seg = x[i:i+W] - x[i:i+W].mean()
        seg = seg * np.hanning(W)
        # NSDF-like autocorrelation
        n = 2*W
        F = np.fft.rfft(seg, n)
        ac = np.fft.irfft(np.abs(F)**2)[:W]
        m = np.cumsum(seg[::-1]**2)[::-1]  # not exact NSDF; use normalized ac
        ac = ac / (ac[0] + 1e-9)
        lmin = int(sr/fmax); lmax = int(sr/fmin)
        lag = lmin + np.argmax(ac[lmin:lmax])
        # parabolic interp
        if 1 <= lag < W-1:
            a, b, c = ac[lag-1], ac[lag], ac[lag+1]
            p = 0.5*(a-c)/(a-2*b+c+1e-12)
        else:
            p = 0
        f0 = sr/(lag+p)
        out.append(((i+W/2)/sr, f0, ac[lag], np.sqrt(np.mean(x[i:i+W]**2))))
    return np.array(out)

sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
tr = f0_track(x, sr)
np.save(sys.argv[2], tr)
# print in cents relative to A440
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
for t, f0, c, r in tr:
    if t0 <= t <= t1:
        cents = 1200*np.log2(f0/440.0)
        print(f'{t:6.2f} f0={f0:7.2f} midi={69+cents/100:6.2f} conf={c:4.2f} rms={r:6.1f}')
