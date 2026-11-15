import sys, importlib, time
sys.path.insert(0, '/home/claude/work/final'); sys.path.insert(1, '/home/claude/work')
import numpy as np
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor

def run_one(args):
    modname, fn, a, b = args
    mod = importlib.import_module(modname)
    sr, d = wavfile.read(fn)
    s, e = mod.detect(d.astype(np.float64), sr)
    return fn, a, b, s, e

if __name__ == '__main__':
    modname = sys.argv[1]
    truth = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/work/cases/truth.csv'
    rows = [l.strip().split(',') for l in open(truth) if l.strip()]
    jobs = [(modname, r[0], float(r[1]), float(r[2])) for r in rows]
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(run_one, jobs))
    by = {}
    for fn, a, b, s, e in res:
        kind = fn.split('/')[-1][0]
        es, ee = abs(s-a), abs(e-b)
        ok = es <= 1.5 and ee <= 1.5
        by.setdefault(kind, []).append(ok)
        if not ok or '-v' in sys.argv:
            print(f'{fn.split("/")[-1]:8s} true {a:6.2f}-{b:6.2f}  pred {s:6.2f}-{e:6.2f}  err {es:5.2f} {ee:5.2f} {"OK" if ok else "FAIL"}')
    for k in sorted(by):
        print(k, f'{sum(by[k])}/{len(by[k])}')
    print('time', time.time()-t0)
