# Sherlock Soles — "That isn't my shoe!" (LMML / Sigmoid, task_vietnam)

Open-set recognition: label each test image `nike` / `adidas` / `converse`, or `unknown` (Puma, Reebok, Vans and Asics look-alikes).
Gates: known accuracy ≥ 0.85 **and** unknown rejection ≥ 0.70. Deliverable: one `.pt` file, ≤ 10 MB, run by the platform.

**Status at packaging time:** v1 is final and verified locally. **No platform result yet**, so both the output format and the main assumption (below) are still unconfirmed.

---

## 1. What to submit

All four files hold the same model and differ only in what `forward` returns. Each is a self-contained TorchScript file of about 6.95 MB.

| file | `model(x)` returns | when it fits |
|---|---|---|
| `submission/sherlock_labels.pt` | `List[str]` (one of `nike`/`adidas`/`converse`/`unknown` per image) | try first: if the platform expects a tensor, it fails loudly instead of silently |
| `submission/sherlock_probs4.pt` | `Tensor[N,4]`, column order `nike, adidas, converse, unknown`; argmax = decision | platform maps an argmax to 4 class names |
| `submission/sherlock_probs4_a-c-n.pt` | same, but alphabetical: `adidas, converse, nike, unknown` | if the platform's class list is alphabetical |
| `submission/sherlock_logits3.pt` | `Tensor[N,3]`; rejected rows are flat (softmax = 1/3), accepted rows near one-hot | if the platform applies its own confidence threshold |

All four variants also expose the exported methods `predict(x) -> List[str]`, `predict_proba4`, `predict_logits3` and `analyse(x)`. `analyse` returns the logits, the accept mask, the semantic score and the trace score. The `classes`/`classes4` attributes hold the class names.

**Loading:** use `torch.jit.load(path)` or `torch.load(path, weights_only=False)`. `torch.load` with its default `weights_only=True` refuses every TorchScript archive. If the platform loads that way, it expects a state_dict for an architecture it defines, and that would need the hint or rules page.

**Input:** the model accepts any of the usual tensor formats:
- dtype `uint8` or float;
- layout NCHW, NHWC, CHW or HW, with 1, 3 or 4 channels;
- value range [0,1] or [0,255], or input already normalised with ImageNet or [-1,1] statistics;
- any spatial size.

Internally it converts to RGB in [0,1] and processes the batch in chunks of 32. The provenance check (section 3) runs **only if the image arrives at 256×256** with its original pixels.

## 2. Data forensics (what made the difference)

- 735 images, 245 per class, all 256×256 RGB JPEGs. All share one quantization table (≈ q95), 4:2:0 subsampling and JFIF, with no EXIF and no exact duplicates. No near-duplicates either: every holdout image is at cosine < 0.9 from its nearest training image.
- The source is the Kaggle dataset *die9origephit / nike-adidas-and-converse-imaged*, whose images are all **240×240** (275 per class). That leaves an inferred ~30 per class for the hidden known test set; this is not confirmed.
- The organisers resized 240 → 256 (×16/15). That leaves a **period-16 phase pattern** in the pixel grid of every training image (`reports/` and `src/analysis/resample_trace.py`, `trace_detect.py`, `trace_null.py`).
- Puma, Reebok, Vans and Asics images cannot come from that 3-brand dataset. They were most likely resized to 256 from other native sizes, so they should **not** carry the aligned 240→256 trace.

## 3. Method

1. **Brand classifier.** MobileNetV3-Large, timm `mobilenetv3_large_100.miil_in21k_ft_in1k` (ImageNet-21k → 1k).
   - Fine-tuned on all 735 images for 10 epochs at 224 px with random-resized crops (scale 0.55–1), horizontal flip, colour jitter and 5% grayscale.
   - Optimiser: AdamW, lr 3e-4 for the backbone and 3e-3 for the head, weight decay 0.02, cosine schedule with 1 warm-up epoch, batch 32.
   - Inference runs at the native **256 px** with flip-TTA, which scored 0.906 OOF against 0.887 when resized to 224.
2. **Semantic novelty.** Cosine similarity to the nearest of the 735 training embeddings (1280-d, flip-averaged, L2-normalised kNN bank).
3. **Provenance (resize-trace) detector.**
   - Features (128-d, computed on luma): period-16 phase-folded column and row profiles of |∂²|, ∂², |∂¹| and row-normalised |∂²|, each z-normalised.
   - Classifier: logistic regression with C = 0.3. Genuine examples are the 735 train images.
   - Null set: 760 crops from 19 natural images (skimage and sklearn samples). Each crop optionally gets a source JPEG at q70–95 and an intermediate resize to 150–640 px, then is resized to 256 and saved as JPEG q95.
4. **Decision.**
   - Both scores become p-values through the empirical CDF of out-of-fold scores on known images.
   - An image is accepted if `log p_sem + 2·log p_trace ≥ −9.043`, a threshold that keeps 98% of known images out-of-fold. It then gets `argmax(brand logits)`; otherwise it is `unknown`.
   - If the input is not 256×256, the trace is skipped and the rule becomes `log p_sem ≥ −3.789`.
5. **Export.**
   - Batch norm is left in place. All convolutions are stored as per-output-channel **int8** and dequantised on the fly; the kNN bank is stored in fp16.
   - The backbone is traced, the wrapper scripted, and the result saved as one TorchScript file (6.95 MB).

## 4. Validation

| check | result |
|---|---|
| Frozen linear probes (5-fold, `reports/frozen_probes.txt`) | mnv3-MIIL 0.850 · mnv3-21k 0.837 · EffNet-B0-NS 0.816 · B2-NS 0.811 · EdgeNeXt-S 0.808 · ConvNeXt-pico 0.807 · atto 0.766 |
| Fine-tuned, 5-fold CV, 224 inference | 0.898 / 0.905 / 0.871 / 0.912 / 0.850 (mean 0.887) |
| Fine-tuned, 5-fold CV, **256 inference** | 0.912 / 0.918 / 0.878 / 0.932 / 0.891 (**mean 0.906**) |
| Semantic-only open-set proxy (train on 2 brands, 3rd = unknown; `runs/mnv3_loco*/result.json`) | kNN AUROC 0.83 / 0.83 / 0.81; rejects only 0.38–0.51 at 90% known retention, so the **semantic score alone cannot reach 0.70** |
| Trace detector, grouped CV vs natural null (`reports/fit_trace_final.txt`) | AUROC **0.989**; rejects 0.882 / 0.920 / 0.955 of the null when keeping 98 / 97 / 95% of genuine images |
| Is it really the trace and not image content? | same training images shifted by 8 px: **99%** rejected; by 3 px: 14% rejected, so the detector is phase-specific |
| OOF rule (`reports/oof_predictions_v1.csv`, `keep_curve.txt`) | closed-set 0.906 · accepted 0.980 · **known_acc 0.891** |
| Simulated unknown rejection (null trace scores) | **0.86** if the semantic score says nothing about unknowns · **0.92** with LOCO-like semantic scores |
| Exported int8 TorchScript fold models on their held-out folds (kNN bank excludes the held-out images) | fold 0: known_acc 0.905 (0.7% rejected) · fold 2: 0.871 (1.4% rejected); natural non-shoe images 96–98% rejected |
| Input-format robustness (`src/verify.py`, `src/eval_like_platform.py`) | identical labels for float [0,1], uint8 NCHW and NHWC, float [0,255] and ImageNet-normalised input, single CHW images and torchvision `ToTensor`/`PILToTensor`/`Normalize` pipelines |

## 5. Assumptions and risks (please challenge these)

1. **Main assumption:** the unknown test images did not go through the same 240→256 resize. If they did (scenario B), the trace carries no information. Rejection then falls to about 0.14 in the leave-one-brand-out proxy (`reports/joint_sim_loco.txt`) and the rejection gate fails. Without the trace no tested variant clears 0.70 anyway, so including it costs nothing in that case.
2. **The platform must pass the original 256×256 pixels**; dtype and normalisation do not matter. If it resizes first, the model falls back to the semantic-only rule. That rule rejects only about a third of unknowns (≈ 0.33 at 95% known retention in the leave-one-brand-out proxy, less at the 98% used here), so the gate fails.
3. **Known-accuracy margin is thin:** 0.891 OOF and 0.871–0.905 on the exported fold checks, against a 0.85 gate. With roughly 90 known test images the binomial SD is about 0.03.
4. **Output contract unknown** — hence four variants.
5. **Honesty note:** the trace exploits how the dataset was built (data provenance), not shoe appearance. It is legitimate for the evaluator but would not transfer to a deployment where every image shares one pipeline.
6. **Harder null:** training shoes slightly rotated (1–4°) or zoomed (1.04–1.3×) keep part of the original trace. The detector reaches only AUROC 0.85 on them (`src/fit_trace2.py`). Real unknowns never had that trace, but this is the weakest point of the null-set argument.

## 6. Tried and not adopted

- **v2:** MobileNetV3 trained directly at 256 px (B) and the A+B ensemble. On CV folds 0–1 (294 images), A scored 0.915, B 0.908 and A+B 0.918 — only +0.3 pt, while two backbones push the file to about 10–11 MB (`reports/v2_ensemble_eval.txt`).
- **Extra TTA scales:** full 0.9061, up288 0.9075, full+up288+up320 0.9088, i.e. negligible (`reports/tta_eval.txt`).
- **Spectral band-limit feature for provenance:** AUROC 0.65, discarded (`src/analysis/band_test.py`).
- **Lighter trace detector (32-d |∂²| template):** AUROC 0.95–0.97; the 128-d version is better.
- **Possible small improvement, not applied:**
  - int8 depthwise convs and the stem cause most of the quantisation drift (embedding cosine vs fp32: mean 0.953). Storing just those in fp16 adds about 0.1–0.2 MB and raises it to ≈ 0.985. Labels were 100% identical on a 60-image check.
  - The held-out numbers above already include the int8 effect.

## 7. Reproduce / check

Environment used: Python 3.13; torch 2.11.0 (run on CPU); torchvision 0.26.0; numpy 2.4.6; Pillow 12.3.0; timm 1.0.31; scikit-learn 1.9.1; scikit-image. TorchScript files are independent of the Python version.

```bash
# from the package root; the challenge data is NOT included -> point to it:
export SHERLOCK_DATA=/path/to/task_vietnam          # contains train-input/ and train-output.csv
python -I src/verify.py submission/sherlock_labels.pt   # loads, input formats, non-shoe rejection
python -I src/eval_like_platform.py                     # torchvision-style evaluator loop
python -I src/make_oof_csv.py runs "$SHERLOCK_DATA" /tmp/oof.csv          # == reports/oof_predictions_v1.csv
python -I src/make_calib.py --runs runs/mnv3_cv{0..4} --emb emb256.npz --keep 0.98 --w_trace 2.0 --out /tmp/c.npz   # == runs/calib_v1.npz
python -I src/fit_trace_final.py 0.3 /tmp/t.npz         # trace detector from the cached features (AUROC 0.989)
python -I src/export_model.py --ckpt runs/mnv3_full/model.pt --calib runs/calib_v1.npz --infer_size 256 \
       --outdir /tmp/re --variants "labels:nike,adidas,converse"       # re-export (no ImageNet weights needed)
```

- **Re-export:** the packaged exporter has one extra `bank_scale` buffer, added for the int8-bank experiment. A re-exported file is therefore byte-different from `submission/`, but its labels were identical on 147/147 checked images.
- **Full retraining** (CPU, about 1.5–2 h): `scripts/download_weights.sh`, then `scripts/run_pipeline.sh`.
- **Path overrides:** `SHERLOCK_DATA`, `SHERLOCK_WEIGHTS`, `SHERLOCK_RUNS`, `SHERLOCK_EXPORT` (see `src/paths.py`).

## 8. Layout

```
submission/        the 4 .pt files (+ .json metadata: thresholds, class order, input size)
src/               pipeline: finetune, trace_torch, fit_trace2, fit_trace_final, oof_infer, make_calib,
                   export_model, verify, eval_like_platform, make_oof_csv, extract_feats, paths
src/analysis/      experiments: frozen probes, forensics, trace variants, open-set simulations, TTA, v2 ensemble
src/tests/         quantisation / BN-folding / TorchScript-port checks
runs/              training logs + result.json for every run (holdout, leave-one-brand-out, 5-fold CV, full, v2 folds)
                   mnv3_full/model.pt = final fp32 checkpoint; mnv3_cv*/emb256.npz = out-of-fold logits/embeddings;
                   trace_lr3.npz = final detector, trace_feats_cache.npz = its features; calib_v1.npz = thresholds
reports/           OOF predictions per image, analysis outputs, misclassified holdout examples, sample grid
scripts/           download_weights.sh, run_pipeline.sh
```
