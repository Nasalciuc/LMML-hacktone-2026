import sys, time
sys.path.insert(0, '/home/claude/work/final')
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
import main as M

T0, T1 = 14.960, 20.965
sr, d = wavfile.read('/home/claude/data/train-input.wav')
x = d.astype(np.float64)

def detect_with(F, tt, drop):
    N = len(tt); hop = tt[1]-tt[0]
    V = [M.gauss_rank(median_filter(F[k], 5, mode='nearest')) for k in sorted(F) if not any(k.startswith(p) for p in drop)]
    V = np.array(V).T; V = V[:, V.std(0) > 1e-3]; V = V - V.mean(0)
    C = np.cov(V.T); C = 0.9*C + 0.1*np.diag(np.diag(C))
    w, U = np.linalg.eigh(C); W = U @ np.diag(1/np.sqrt(np.maximum(w, 1e-6))) @ U.T
    _, a, b = M.best_block(V @ W, int(round(1.5/hop)), min(int(0.5*N), int(round(40/hop))))
    return tt[a]-hop/2, tt[b-1]+hop/2, V.shape[1]

tt, F = M.compute_features(x, sr)
print('n features', len(F))
groups = {
 'none': [],
 'low-freq (lf_, subf0, infra, dc)': ['lf_', 'subf0', 'infra', 'dc'],
 'pitch (f0log, tune, f0_jit)': ['f0log', 'tune', 'f0_jit'],
 'harmonics (h*, even_odd, hslope, centroid)': ['h2', 'h3', 'h4', 'h5', 'even_odd', 'hslope', 'centroid'],
 'mfcc': ['mfcc'],
 'low-freq + pitch': ['lf_', 'subf0', 'infra', 'dc', 'f0log', 'tune', 'f0_jit'],
 'low-freq + pitch + harmonics': ['lf_', 'subf0', 'infra', 'dc', 'f0log', 'tune', 'f0_jit', 'h2', 'h3', 'h4', 'h5', 'even_odd', 'hslope', 'centroid'],
 'low-freq + pitch + harmonics + mfcc': ['lf_', 'subf0', 'infra', 'dc', 'f0log', 'tune', 'f0_jit', 'h2', 'h3', 'h4', 'h5', 'even_odd', 'hslope', 'centroid', 'mfcc'],
}
for g, drop in groups.items():
    s, e, nf = detect_with(F, tt, drop)
    ok = abs(s-T0) <= 1.5 and abs(e-T1) <= 1.5
    print(f'drop {g:45s} left={nf:2d} -> {s:6.2f} {e:6.2f} {"OK" if ok else "FAIL"}')

# awkward lengths (prime sample counts) and timing
def is_prime(n):
    if n % 2 == 0: return False
    i = 3
    while i*i <= n:
        if n % i == 0: return False
        i += 2
    return True
n = len(x)
while not is_prime(n): n -= 1
t = time.time(); s, e = M.detect(x[:n], sr); print(f'prime length {n}: {s:.3f} {e:.3f} in {time.time()-t:.2f}s')
# determinism
r = [M.detect(x, sr) for _ in range(3)]
print('deterministic:', len(set(r)) == 1, r[0])
# time-reversed and inverted polarity
s, e = M.detect(-x, sr); print(f'inverted polarity: {s:.3f} {e:.3f}')
s, e = M.detect(x[::-1].copy(), sr); dur = len(x)/sr; print(f'reversed: {dur-e:.3f} {dur-s:.3f} (mapped back)')
# splice moved to edges by rotating the file
for shift in (14.0, 36.0):
    k = int(shift*sr); y = np.roll(x, -k) if shift < 20 else np.roll(x, len(x)-k)
    ts = (T0 - shift) % dur; te = (T1 - shift) % dur
    s, e = M.detect(y, sr); print(f'rotated by {shift}: true {ts:.2f}-{te:.2f} pred {s:.2f}-{e:.2f}')
