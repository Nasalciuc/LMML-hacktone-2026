"""Print the anonymous photos as a week x counter grid with the attribution and margins."""
import csv
import sys

import numpy as np

hs = ['amayadori', 'jihanki', 'rojiura', 'shuuden', 'toorisugari', 'yakin']
f = sys.argv[1] if len(sys.argv) > 1 else 'attribution.csv'
rows = list(csv.DictReader(open(f)))
B = np.full((5, 29), '?', dtype=object)
G = np.zeros((5, 29))
Zb = np.zeros((5, 29))
for r in rows:
    w = int(r['file'][1:3]) - 8
    n = int(r['file'][4:7]) - 1
    z = np.array([float(r['z_' + h]) for h in hs])
    G[w, n] = z[4] - np.delete(z, 4).max()
    Zb[w, n] = float(r['z_margin'])
    B[w, n] = r['photographer'][0].upper()
print('     ' + ''.join(str((i + 1) // 10) for i in range(29)))
print('     ' + ''.join(str((i + 1) % 10) for i in range(29)))
for w in range(5):
    print('w%02d ' % (w + 8) + ' ' + ''.join(B[w]))
print()
for w in range(5):
    print('w%02d ' % (w + 8) + ' ' + ''.join('#' if v > 0 else '.' for v in G[w]))
print('\ntoorisugari margin')
for w in range(5):
    print('w%02d ' % (w + 8) + ' '.join(f'{v:+4.0f}' for v in G[w]))
print('\nbest margin')
for w in range(5):
    print('w%02d ' % (w + 8) + ' '.join(f'{v:4.0f}' for v in Zb[w]))
