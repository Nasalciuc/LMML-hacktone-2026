"""All per-pixel features for one image (page or exhibit)."""
import numpy as np, cv2
from segment import stamp_masks
from noise_features import compute as noise_compute, add_context
from nlf import nlf_features
from dq import dq_features


def all_features(img, stamps=None):
    st = stamp_masks(img) if stamps is None else stamps
    F = noise_compute(img)
    F, sid = add_context(F, st, img.shape[:2])
    F.update(nlf_features(img, st))
    G, M, info = dq_features(img, st, prefix='dq', clip_hi=254, clip_lo=1, cf_llr=0.05)
    F.update(G)
    G2, M2, info2 = dq_features(img, st, prefix='dr', clip_hi=255, clip_lo=0, cf_llr=0.3)
    F.update(G2)
    names = sorted(F.keys())
    return names, F, st, sid, info
