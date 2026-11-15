#!/bin/sh
# Full reproduction of the v1 submission (CPU, 2 threads: roughly 1.5-2 h).
# Prereqs: python 3.10-3.13, pip install torch==2.11.0 torchvision==0.26.0 timm scikit-learn scikit-image pillow numpy
#          data in data/task_vietnam/{train-input/, train-output.csv}; weights via scripts/download_weights.sh
set -e
cd "$(dirname "$0")/.."
PY="python -I"

# 1) provenance (resize-trace) detector: natural-image null set + 128-d phase-folded features, then final LR (C=0.3)
$PY src/fit_trace2.py data/task_vietnam runs/trace_lr2.npz 0.3     # also writes runs/trace_feats_cache.npz
$PY src/fit_trace_final.py 0.3 runs/trace_lr3.npz

# 2) 5-fold CV of the brand classifier (MobileNetV3-L MIIL, trained at 224, 10 epochs) -> out-of-fold scores
for k in 0 1 2 3 4; do
  $PY src/finetune.py --name mnv3_miil1k --split kfold:$k --epochs 10 --lr 3e-4 --out runs/mnv3_cv$k
done
$PY src/oof_infer.py 256 runs/mnv3_cv0 runs/mnv3_cv1 runs/mnv3_cv2 runs/mnv3_cv3 runs/mnv3_cv4   # OOF at native 256 px

# 3) calibration of the accept/reject rule (Fisher combination, trace weight 2, keep 98% of knowns)
$PY src/make_calib.py --runs runs/mnv3_cv0 runs/mnv3_cv1 runs/mnv3_cv2 runs/mnv3_cv3 runs/mnv3_cv4 \
    --emb emb256.npz --keep 0.98 --w_trace 2.0 --out runs/calib_v1.npz

# 4) final model on all 735 images, export the four output variants, verify
$PY src/finetune.py --name mnv3_miil1k --split full --epochs 10 --lr 3e-4 --out runs/mnv3_full
$PY src/export_model.py --ckpt runs/mnv3_full/model.pt --calib runs/calib_v1.npz --infer_size 256 --outdir export \
    --variants "probs4:nike,adidas,converse;probs4:adidas,converse,nike;labels:nike,adidas,converse;logits3:nike,adidas,converse"
$PY src/verify.py export/sherlock_labels.pt
$PY src/make_oof_csv.py runs data/task_vietnam reports/oof_predictions_v1.csv
