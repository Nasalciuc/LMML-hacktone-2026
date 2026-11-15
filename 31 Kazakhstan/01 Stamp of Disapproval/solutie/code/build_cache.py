"""Compute and cache features for exhibits (full maps) and album pages (stride-2 stamp pixels)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from allfeatures import all_features

def run(data, cache):
    os.makedirs(cache, exist_ok=True)
    for i in range(1, 31):
        fn = f'{cache}/ex{i:02d}.npz'
        if os.path.exists(fn): continue
        im = cv2.imread(f'{data}/exhibits/exhibit_{i:02d}.jpg')
        m = cv2.imread(f'{data}/exhibits/exhibit_{i:02d}_mask.png', 0) > 127
        names, F, st, sid, info = all_features(im)
        stamp = sid >= 0
        dist = cv2.distanceTransform(stamp.astype(np.uint8), cv2.DIST_L2, 5)
        np.savez_compressed(fn, X=np.stack([F[n] for n in names], -1).astype(np.float16), mask=m, stamp=stamp,
                            dist=dist.astype(np.float16), names=np.array(names))
        print('exhibit', i, flush=True)
    meta = {}
    for a in sorted(os.listdir(f'{data}/albums')):
        for f in sorted(os.listdir(f'{data}/albums/{a}')):
            if not f.endswith('.jpg'): continue
            key = f'{a}/{f[:-4]}'
            fn = f'{cache}/{a}_{f[:-4]}.npz'
            if os.path.exists(fn): continue
            t = time.time()
            im = cv2.imread(f'{data}/albums/{a}/{f}')
            names, F, st, sid, info = all_features(im)
            stamp = sid >= 0
            dist = cv2.distanceTransform(stamp.astype(np.uint8), cv2.DIST_L2, 5)
            cand = stamp & (dist > 6)
            sub = np.zeros_like(cand); sub[::2, ::2] = True
            idx = np.argwhere(cand & sub).astype(np.int16)
            X = np.stack([F[n][idx[:, 0], idx[:, 1]] for n in names], -1).astype(np.float16)
            np.savez(fn, X=X, idx=idx, sid=sid[idx[:, 0], idx[:, 1]].astype(np.int8),
                     dist=dist[idx[:, 0], idx[:, 1]].astype(np.float16), names=np.array(names),
                     stamps=np.array([bb for bb, _ in st]))
            print(key, X.shape, '%.1fs' % (time.time() - t), flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
