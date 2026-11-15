"""Independent check of the Sherlock Soles model on real photos of other brands.

    python -I soles_test.py <model.pt> <task_vietnam folder> <hf_sneakers folder> <out.json>

Uses the 735 task photos and the local copy of ipogorelov/sneakers (native-size JPEGs with a
brand label). Nothing here comes from the package under test except the .pt file.
"""
import collections
import csv
import glob
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

MODEL, DATA, HF, OUT = sys.argv[1:5]
N_PER = int(sys.argv[5]) if len(sys.argv) > 5 else 150
UNKNOWN = ["puma", "reebok", "vans", "asics"]
KNOWN = ["nike", "adidas", "converse"]
torch.set_num_threads(8)
rng = np.random.default_rng(0)
m = torch.jit.load(MODEL, map_location="cpu").eval()
classes = list(m.classes)
sem_cal = m.sem_cal.numpy()
tr_cal = m.tr_cal.numpy()
THR_J, THR_S, W = float(m.thr_joint), float(m.thr_sem), float(m.w_trace)
R = {"model": os.path.basename(MODEL), "thr_joint": THR_J, "thr_sem": THR_S, "w_trace": W}


def pval(cal, s):
    return (np.searchsorted(cal, s, side="right") + 1.0) / (len(cal) + 2.0)


def analyse(x):
    """x: N,256,256,3 uint8 -> label, brand guess, accepted, semantic score, trace score"""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous()
    lg, ac, ss, st = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(t), 64):
            a, b, c, d = m.analyse(t[i:i + 64])
            lg.append(a), ac.append(b), ss.append(c), st.append(d)
    lg, ac, ss, st = torch.cat(lg), torch.cat(ac).numpy().astype(bool), torch.cat(ss).numpy(), torch.cat(st).numpy()
    guess = np.array([classes[i] for i in lg.argmax(1).tolist()])
    return np.where(ac, guess, "unknown"), guess, ac, ss, st


def trace_only(x):
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        return torch.cat([m.trace_score(t[i:i + 128]) for i in range(0, len(t), 128)]).numpy()


def jpeg(a, q):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))


PIL_K = {"pil_nearest": Image.NEAREST, "pil_bilinear": Image.BILINEAR, "pil_bicubic": Image.BICUBIC,
         "pil_lanczos": Image.LANCZOS, "pil_box": Image.BOX}
CV_K = {"cv_linear": cv2.INTER_LINEAR, "cv_cubic": cv2.INTER_CUBIC, "cv_area": cv2.INTER_AREA,
        "cv_lanczos": cv2.INTER_LANCZOS4, "cv_nearest": cv2.INTER_NEAREST}


def resize(a, s, k):
    if k in PIL_K:
        return np.asarray(Image.fromarray(a).resize((s, s), PIL_K[k]))
    return cv2.resize(a, (s, s), interpolation=CV_K[k])


def direct(a, k):
    return jpeg(resize(a, 256, k), 95)


def via(a, s, k2, k1="pil_lanczos", q1=90):
    b = resize(a, s, k1)
    if q1:
        b = jpeg(b, q1)
    if s != 256:
        b = resize(b, 256, k2)
    return jpeg(b, 95)


def stack(imgs, f):
    return np.stack([f(a) for a in imgs])


def show(name, d):
    print(name, json.dumps(d), flush=True)
    R[name] = d


# ---------------------------------------------------------------- 1. task photos
rows = list(csv.DictReader(open(os.path.join(DATA, "train-output.csv"))))
tr = np.stack([np.asarray(Image.open(os.path.join(DATA, "train-input", r["filename"])).convert("RGB")) for r in rows])
ty = np.array([r["label"] for r in rows])
t0 = time.time()
lab, guess, acc, ss, st = analyse(tr)
dt = (time.time() - t0) / len(tr)
tr_q = np.percentile(st, [1, 2, 5, 50, 95]).round(2).tolist()
show("train_as_is", {"n": len(tr), "accepted": round(float(acc.mean()), 4), "label_correct": round(float((lab == ty).mean()), 4),
                     "brand_guess_correct_in_sample": round(float((guess == ty).mean()), 4),
                     "trace_score_pct_1_2_5_50_95": tr_q, "ms_per_image": round(dt * 1000, 1)})
for sh in (3, 8, 16):
    x = np.roll(tr, (sh, sh), axis=(1, 2))
    l2, g2, a2, s2, t2 = analyse(x)
    show(f"train_shifted_{sh}px", {"rejected": round(float((~a2).mean()), 4), "trace_score_median": round(float(np.median(t2)), 2)})
x = stack(tr, lambda a: jpeg(a, 95))
l2, g2, a2, s2, t2 = analyse(x)
show("train_resaved_jpeg95", {"rejected": round(float((~a2).mean()), 4), "label_correct": round(float((l2 == ty).mean()), 4),
                              "trace_score_median": round(float(np.median(t2)), 2)})
x = stack(tr, lambda a: via(a, 224, "pil_bilinear", "pil_bilinear", 0))
l2, g2, a2, s2, t2 = analyse(x)
show("train_down_to_224_and_back", {"rejected": round(float((~a2).mean()), 4), "label_correct": round(float((l2 == ty).mean()), 4)})


# ---------------------------------------------------------------- 2. other-source photos
def key(a):
    g = np.asarray(Image.fromarray(a).convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


tk = np.stack([key(a) for a in tr] + [key(a[:, ::-1]) for a in tr])
files = sorted(glob.glob(os.path.join(HF, "*.parquet")))
files = [files[i] for i in rng.permutation(len(files))]
imgs = collections.defaultdict(list)
sizes = collections.Counter()
dups = 0
per_file = max(4, N_PER // 12)
for f in files:
    if all(len(imgs[b]) >= N_PER for b in UNKNOWN + KNOWN):
        break
    t = pq.read_table(f, columns=["brand", "image"])
    br = [b.strip().lower() for b in t["brand"].to_pylist()]
    got = collections.Counter()
    for i in rng.permutation(len(br)):
        b = br[i]
        if b not in UNKNOWN + KNOWN or len(imgs[b]) >= N_PER or got[b] >= per_file:
            continue
        try:
            im = Image.open(io.BytesIO(t["image"][int(i)].as_py())).convert("RGB")
        except Exception:
            continue
        if min(im.size) < 170:
            continue
        a = np.asarray(im)
        if b in KNOWN and float((tk @ key(a)).max()) / 1024.0 > 0.95:
            dups += 1
            continue
        imgs[b].append(a)
        got[b] += 1
        sizes[im.size] += 1
show("other_source_sample", {"per_brand": {b: len(imgs[b]) for b in UNKNOWN + KNOWN}, "near_duplicates_of_task_photos_dropped": dups,
                             "native_sizes_top": [[list(k), v] for k, v in sizes.most_common(8)]})
U = [a for b in UNKNOWN for a in imgs[b]]
Ub = np.array([b for b in UNKNOWN for _ in imgs[b]])
K = [a for b in KNOWN for a in imgs[b]]
Kb = np.array([b for b in KNOWN for _ in imgs[b]])

# 2a. which resize kernel leaves the same trace as the task photos? (trace score only)
thr2 = tr_q[1]
kern = {}
for k2 in list(PIL_K) + list(CV_K):
    s = trace_only(stack(U[:240], lambda a: via(a, 240, k2)))
    kern[k2] = {"trace_score_median": round(float(np.median(s)), 2), "share_above_task_2nd_pct": round(float((s >= thr2).mean()), 3)}
show("kernel_240_to_256", kern)
best = max(kern, key=lambda k: kern[k]["trace_score_median"])
R["best_kernel"] = best
print("best kernel:", best, flush=True)


def scen(name, fU, fK=None):
    xu = stack(U, fU)
    lab, guess, acc, ss, st = analyse(xu)
    ps, pt = pval(sem_cal, ss), pval(tr_cal, st)
    d = {"unknown_n": len(U), "unknown_rejected": round(float((lab == "unknown").mean()), 4),
         "per_brand": {b: round(float((lab[Ub == b] == "unknown").mean()), 3) for b in UNKNOWN},
         "rejected_by_semantic_rule_alone": round(float((np.log(ps) < THR_S).mean()), 4),
         "trace_looks_genuine(p>0.05)": round(float((pt > 0.05).mean()), 4)}
    if fK is not None:
        xk = stack(K, fK)
        lab, guess, acc, ss, st = analyse(xk)
        d.update({"known_n": len(K), "known_accepted": round(float(acc.mean()), 4),
                  "known_accuracy_as_scored": round(float((lab == Kb).mean()), 4),
                  "known_brand_guess_correct": round(float((guess == Kb).mean()), 4),
                  "known_per_brand_as_scored": {b: round(float((lab[Kb == b] == b).mean()), 3) for b in KNOWN}})
    show(name, d)


# 2b. unknown brands resized straight to 256 (the package's main assumption)
for k in ("pil_bilinear", "pil_bicubic", "pil_lanczos", "cv_area", "cv_linear"):
    scen(f"A_direct_{k}", lambda a, k=k: direct(a, k))
# 2c. everything through 240 -> 256 with the kernel that matches the task photos (worst case for unknowns,
#     realistic case for knowns from a new source)
scen(f"B_via240_{best}", lambda a: via(a, 240, best), lambda a: via(a, 240, best))
scen(f"B_via240_{best}_nojpeg240", lambda a: via(a, 240, best, q1=0), lambda a: via(a, 240, best, q1=0))
# 2d. unknown brands from other fixed source sizes, same final kernel
for s in (128, 160, 200, 224, 256, 288, 300, 320, 384, 400, 480, 512):
    scen(f"C_source_{s}_{best}", lambda a, s=s: via(a, s, best))

# ---------------------------------------------------------------- 3. one image per call, several input formats
sel = np.arange(0, len(tr), 4)
ref = lab_ref = analyse(tr[sel])[0]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
fm = {"float01": lambda t: t.float() / 255.0, "imagenet_norm": lambda t: (t.float() / 255.0 - mean) / std,
      "minus1_plus1": lambda t: t.float() / 127.5 - 1.0, "float_0_255": lambda t: t.float(), "uint8": lambda t: t}
single = {}
with torch.no_grad():
    for name, f in fm.items():
        out = []
        for i in sel:
            t = torch.from_numpy(tr[i]).permute(2, 0, 1).unsqueeze(0)
            out.append(m.predict(f(t))[0])
        single[name] = round(float((np.array(out) == ref).mean()), 4)
show("one_image_per_call_agreement_with_batch", {"n": int(len(sel)), **single})
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT)
