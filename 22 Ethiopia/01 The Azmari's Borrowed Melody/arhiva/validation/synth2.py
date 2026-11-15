import sys, os, json
sys.path.insert(0, '/home/claude/work')
import numpy as np
from scipy import signal
from scipy.io import wavfile
import synth
from synth import make_take, chain, random_chain, splice, SR

def make_case(seed, scenario):
    rng = np.random.default_rng(seed)
    total = rng.uniform(30, 90)
    L = rng.uniform(2.0, 12.0)
    instA = rng.choice(['masenqo', 'washint'])
    if scenario == 'diff':
        instB = 'washint' if instA == 'masenqo' else 'masenqo'
    else:
        instB = instA
    cA = random_chain(rng)
    cB = random_chain(rng) if scenario != 'samechain' else cA
    tA = make_take(rng, instA, total); tB = make_take(rng, instB, L + 2)
    A = chain(rng, tA, cA); B = chain(rng, tB, cB)
    st = rng.uniform(0, 1.5)
    ins = B[int(st*SR):int((st+L)*SR)]
    pos = int(rng.uniform(0.2, total - L - 0.2)*SR)
    y = splice(A, ins, pos, rng.choice([0.01, 0.03, 0.06, 0.12, 0.3]))
    out_sr = int(rng.choice([16000, 22050, 22050, 44100]))
    if out_sr != SR:
        y = signal.resample_poly(y, out_sr, SR) if out_sr % 50 == 0 else signal.resample(y, int(len(y)*out_sr/SR))
    y = y/np.max(np.abs(y))*rng.uniform(1500, 20000)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16), out_sr, pos/SR, (pos+len(ins))/SR

if __name__ == '__main__':
    n = int(sys.argv[1]); scen = sys.argv[2].split(',')
    d = '/home/claude/work/syn2'
    rows = []
    for sc in scen:
        for s in range(n):
            y, sr, a, b = make_case(10007*len(sc) + 31*s + 5, sc)
            fn = f'{d}/{sc}_{s:02d}.wav'
            wavfile.write(fn, sr, y)
            rows.append(f'{fn},{a:.3f},{b:.3f}')
    open(f'{d}/truth.csv', 'w').write('\n'.join(rows)+'\n')
    print(len(rows))
