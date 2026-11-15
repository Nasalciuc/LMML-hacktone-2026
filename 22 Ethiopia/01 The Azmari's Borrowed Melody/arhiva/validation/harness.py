import numpy as np
from scipy.io import wavfile
from scipy import signal

SR = 22050
_sr, _d = wavfile.read('/home/claude/data/train-input.wav')
X = _d.astype(np.float64)
T0, T1 = 14.960, 20.965
ORIG = [X[:int((T0-0.08)*SR)], X[int((T1+0.08)*SR):]]
WASH = X[int((T0+0.06)*SR):int((T1-0.06)*SR)]

def envelope(x, cutoff=20):
    sos = signal.butter(2, cutoff, fs=SR, output='sos')
    return np.sqrt(np.maximum(signal.sosfiltfilt(sos, x**2), 0))

def cut_points(x):
    e = envelope(x)
    # local minima of envelope separated by >= 0.15 s
    pk, _ = signal.find_peaks(-e, distance=int(0.15*SR), prominence=50)
    return pk

def xfade_concat(chunks, fade=0.02):
    n = int(fade*SR)
    out = chunks[0].copy()
    for c in chunks[1:]:
        w = np.linspace(0, 1, n)
        out[-n:] = out[-n:]*(1-w) + c[:n]*w
        out = np.concatenate([out, c[n:]])
    return out

def resample_shift(x, semitones):
    # pitch shift by resampling (changes duration)
    ratio = 2**(semitones/12)
    n_out = int(round(len(x)/ratio))
    return signal.resample_poly(x, n_out, len(x)) if False else signal.resample(x, n_out)

def rearrange(sources, total_len, rng, min_chunk=2.5, max_chunk=7.0):
    cps = [cut_points(s) for s in sources]
    chunks = []; L = 0
    while L < total_len*SR:
        k = rng.integers(len(sources))
        s, cp = sources[k], cps[k]
        if len(cp) < 3: continue
        i = rng.integers(len(cp)-1)
        dur = rng.uniform(min_chunk, max_chunk)
        j = np.searchsorted(cp, cp[i] + dur*SR)
        if j >= len(cp): continue
        c = s[cp[i]:cp[j]]
        chunks.append(c); L += len(c)
    out = xfade_concat(chunks)
    return out[:int(total_len*SR)]

def splice(base, ins, pos, fade=0.04):
    n = int(fade*SR)
    L = len(ins)
    seg_base = base[pos:pos+L]
    g = np.sqrt(np.mean(seg_base**2)/np.mean(ins**2))
    ins = ins*g
    out = base.copy()
    w = np.ones(L); w[:n] = np.linspace(0, 1, n); w[-n:] = np.linspace(1, 0, n)
    out[pos:pos+L] = base[pos:pos+L]*(1-w) + ins*w
    return out

def make_case(seed, kind):
    rng = np.random.default_rng(seed)
    total = rng.uniform(50, 60)
    if kind == 'A':     # washint (as is) into rearranged original
        base = rearrange(ORIG, total, rng)
        L = rng.uniform(2.5, len(WASH)/SR - 0.05)
        st = rng.uniform(0, len(WASH)/SR - L)
        ins = WASH[int(st*SR):int((st+L)*SR)]
    elif kind == 'B':   # washint shifted down into original register
        base = rearrange(ORIG, total, rng)
        sh = rng.uniform(-12, -5)
        w = resample_shift(WASH, sh)
        L = rng.uniform(3.0, min(9.0, len(w)/SR - 0.05))
        st = rng.uniform(0, len(w)/SR - L)
        ins = w[int(st*SR):int((st+L)*SR)]
    elif kind == 'C':   # original instrument, shifted register
        base = rearrange(ORIG, total, rng)
        sh = rng.choice([-7, -5, -4, -3, 3, 4, 5, 7]) + rng.uniform(-0.3, 0.3)
        src = ORIG[rng.integers(2)]
        st = rng.uniform(0, len(src)/SR - 12)
        chunk = src[int(st*SR):int((st+12)*SR)]
        w = resample_shift(chunk, sh)
        L = rng.uniform(3.0, min(8.0, len(w)/SR - 0.1))
        ins = w[:int(L*SR)]
    elif kind == 'D':   # original (masenqo) inserted into washint-based base
        srcs = [resample_shift(WASH, s) for s in (-7, -5, -2, 0, 2, -9)]
        base = rearrange(srcs, total, rng, 1.5, 4.0)
        src = ORIG[rng.integers(2)]
        sh = rng.uniform(2, 9)
        st = rng.uniform(0, len(src)/SR - 12)
        w = resample_shift(src[int(st*SR):int((st+12)*SR)], sh)
        L = rng.uniform(3.0, min(8.0, len(w)/SR - 0.1))
        ins = w[:int(L*SR)]
    pos = int(rng.uniform(1.0, total - L - 1.0)*SR)
    y = splice(base, ins, pos)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y, pos/SR, (pos+len(ins))/SR

if __name__ == '__main__':
    import sys, os
    os.makedirs('/home/claude/work/cases', exist_ok=True)
    rows = []
    for kind in 'ABCD':
        for s in range(int(sys.argv[1])):
            y, a, b = make_case(1000*ord(kind)+s, kind)
            fn = f'/home/claude/work/cases/{kind}{s:02d}.wav'
            wavfile.write(fn, SR, y)
            rows.append(f'{fn},{a:.3f},{b:.3f}')
    open('/home/claude/work/cases/truth.csv', 'w').write('\n'.join(rows)+'\n')
    print(len(rows), 'cases')
