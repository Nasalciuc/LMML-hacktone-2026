import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal

sr, d = wavfile.read(sys.argv[1])
x = d.astype(np.float64)
T0, T1 = 14.960, 20.965
win = int(0.25 * sr)
n = len(x) // win
rows = []
for i in range(n):
    seg = x[i*win:(i+1)*win]
    t = (i + 0.5) * win / sr
    f, P = signal.periodogram(seg * np.hanning(len(seg)), sr)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return 10*np.log10(P[m].mean() + 1e-20)
    rows.append(dict(
        t=t,
        dc=seg.mean(),
        rms=np.sqrt(np.mean(seg**2)),
        b0_20=band(0.5, 20), b20_45=band(20, 45), b45_55=band(45, 55), b55_65=band(55, 65), b65_150=band(65, 150),
        b4k_6k=band(4000, 6000), b6k_8k=band(6000, 8000), b8k_10k=band(8000, 10000), b10k=band(10000, 11025),
        odd=np.mean(seg.astype(int) % 2), 
    ))
keys = [k for k in rows[0] if k != 't']
ins = np.array([T0 <= r['t'] <= T1 for r in rows])
# exclude windows straddling boundaries
edge = np.array([abs(r['t']-T0) < 0.25 or abs(r['t']-T1) < 0.25 for r in rows])
print(f"{'feat':10s} {'in_mean':>9s} {'in_std':>8s} {'out_mean':>9s} {'out_std':>8s} {'d-prime':>8s}")
for k in keys:
    v = np.array([r[k] for r in rows])
    a = v[ins & ~edge]; b = v[~ins & ~edge]
    dp = (a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var())+1e-12)
    print(f"{k:10s} {a.mean():9.3f} {a.std():8.3f} {b.mean():9.3f} {b.std():8.3f} {dp:8.2f}")
