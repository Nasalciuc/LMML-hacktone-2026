"""Validation of the strong-DQ decision rule on the exhibits: when the most DQ-incompatible
33x33 window of a stamp sums to < -50 (log-likelihood ratio), is it inside the altered region?"""
import sys
import numpy as np, cv2
from segment import stamp_masks
from dq import dq_maps

def main(data, thr=-50.0):
    fired = correct = 0
    for i in range(1, 31):
        im = cv2.imread(f'{data}/exhibits/exhibit_{i:02d}.jpg')
        m = cv2.imread(f'{data}/exhibits/exhibit_{i:02d}_mask.png', 0) > 127
        st = stamp_masks(im)
        M, info = dq_maps(im, st)
        keep = np.zeros(m.shape, bool)
        for _, s in st:
            keep |= cv2.distanceTransform(s.astype(np.uint8), cv2.DIST_L2, 5) > 12
        Ls = cv2.boxFilter(M['L'] * keep, -1, (33, 33), normalize=False)
        cy, cx = np.unravel_index(np.argmin(Ls), Ls.shape)
        hit = (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[cy, cx]
        if Ls[cy, cx] < thr:
            fired += 1; correct += bool(hit)
        print('exhibit %02d  min %.1f at (%d,%d)  %s%s' % (i, Ls[cy, cx], cx, cy, 'inside mask' if hit else 'outside', '  [rule fires]' if Ls[cy, cx] < thr else ''))
    print('rule fired on %d exhibits, correct %d' % (fired, correct))

if __name__ == '__main__':
    main(sys.argv[1])
