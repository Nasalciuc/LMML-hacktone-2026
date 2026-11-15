import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os, sys, csv, hashlib, collections
import numpy as np
from PIL import Image

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
lab = {r['filename']: r['label'] for r in rows}
img_dir = os.path.join(D, 'train-input')

qsig = collections.Counter()
qsig_by = collections.defaultdict(collections.Counter)
sizes = []
md5s = collections.defaultdict(list)
info_keys = collections.Counter()
subs = collections.Counter()
progressive = collections.Counter()
for f, l in lab.items():
    p = os.path.join(img_dir, f)
    raw = open(p, 'rb').read()
    md5s[hashlib.md5(raw).hexdigest()].append(f)
    sizes.append(len(raw))
    im = Image.open(p)
    q = getattr(im, 'quantization', None)
    sig = tuple((k, tuple(v[:8])) for k, v in sorted(q.items())) if q else None
    qsig[sig] += 1
    qsig_by[l][sig] += 1
    for k in im.info.keys():
        info_keys[k] += 1
    try:
        from PIL import JpegImagePlugin
        subs[JpegImagePlugin.get_sampling(im)] += 1
    except Exception as e:
        subs['err'] += 1
    progressive[im.info.get('progressive', 0) or im.info.get('progression', 0)] += 1

print('n images', len(lab))
print('distinct quant tables:', len(qsig))
for s, c in qsig.most_common(10):
    print(c, s if s is None else [ (k, v[:4]) for k, v in s])
print('per class quant table counts:', {k: len(v) for k, v in qsig_by.items()})
print('info keys:', info_keys.most_common())
print('subsampling:', subs)
print('progressive:', progressive)
print('file size: min %d median %d max %d' % (min(sizes), int(np.median(sizes)), max(sizes)))
dups = [v for v in md5s.values() if len(v) > 1]
print('exact duplicate groups:', len(dups), dups[:5])
