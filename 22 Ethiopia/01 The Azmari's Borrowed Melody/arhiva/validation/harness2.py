import numpy as np, sys, os
sys.path.insert(0, '/home/claude/work')
from scipy.io import wavfile
from harness import ORIG, WASH, SR, rearrange, resample_shift, xfade_concat

def splice(base, ins, pos, fade):
    n = max(2, int(fade*SR))
    L = len(ins)
    g = np.sqrt(np.mean(base[pos:pos+L]**2)/np.mean(ins**2))
    ins = ins*g
    out = base.copy()
    w = np.ones(L); w[:n] = np.linspace(0, 1, n); w[-n:] = np.linspace(1, 0, n)
    out[pos:pos+L] = base[pos:pos+L]*(1-w) + ins*w
    return out

def wash_long(rng, total):
    srcs = [resample_shift(WASH, s) for s in (-7, -5, -3, -2, 0, 2, -9, -12)]
    return rearrange(srcs, total, rng, 1.0, 3.5)

def make_case(seed, kind):
    rng = np.random.default_rng(seed)
    total = rng.uniform(45, 62)
    fade = rng.choice([0.005, 0.02, 0.05, 0.1, 0.25])
    edge = None
    if kind in 'AEF':
        base = rearrange(ORIG, total, rng)
        if kind == 'F' and rng.random() < 0.5:
            L = rng.uniform(1.8, 3.0)
            w = WASH
        elif kind == 'F':
            L = rng.uniform(8.0, 14.0)
            w = wash_long(rng, 16)
        else:
            sh = rng.uniform(-12, 3)
            w = resample_shift(WASH, sh)
            L = rng.uniform(2.5, min(10.0, len(w)/SR - 0.1))
        st = rng.uniform(0, len(w)/SR - L)
        ins = w[int(st*SR):int((st+L)*SR)]
        if kind == 'E': edge = rng.choice(['start', 'end'])
    elif kind == 'H':   # washint into washint base, different register
        base = wash_long(rng, total)
        sh = rng.choice([-10, -8, 4, 5, 7]) + rng.uniform(-0.4, 0.4)
        w = resample_shift(WASH, sh)
        L = rng.uniform(2.5, min(8.0, len(w)/SR - 0.1))
        st = rng.uniform(0, len(w)/SR - L)
        ins = w[int(st*SR):int((st+L)*SR)]
    elif kind == 'D':
        base = wash_long(rng, total)
        src = ORIG[rng.integers(2)]
        sh = rng.uniform(-3, 9)
        st = rng.uniform(0, max(0.1, len(src)/SR - 14.5))
        w = resample_shift(src[int(st*SR):int((st+14.5)*SR)], sh)
        L = rng.uniform(2.5, min(10.0, len(w)/SR - 0.1))
        ins = w[:int(L*SR)]
    if edge == 'start':
        pos = int(rng.uniform(0.2, 2.0)*SR)
    elif edge == 'end':
        pos = int((total - L - rng.uniform(0.2, 2.0))*SR)
    else:
        pos = int(rng.uniform(1.0, total - L - 1.0)*SR)
    y = splice(base, ins, pos, fade)
    if rng.random() < 0.3:
        y = y + rng.normal(0, rng.uniform(0.5, 4.0), len(y))
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y, pos/SR, (pos+len(ins))/SR

if __name__ == '__main__':
    n = int(sys.argv[1]); kinds = sys.argv[2]
    d = '/home/claude/work/cases2'
    os.makedirs(d, exist_ok=True)
    rows = []
    for kind in kinds:
        for s in range(n):
            y, a, b = make_case(7777*ord(kind)+s, kind)
            fn = f'{d}/{kind}{s:02d}.wav'
            wavfile.write(fn, SR, y)
            rows.append(f'{fn},{a:.3f},{b:.3f}')
    open(f'{d}/truth.csv', 'w').write('\n'.join(rows)+'\n')
    print(len(rows))
