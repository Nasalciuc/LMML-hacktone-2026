# Laos - Bad BOMB

## Results

`test-output.txt`: 28 image names, one per line.

No labels exist, so no score. Visual check of every pick: 26 of the 28
show a clear defect, in three repeated types:

| type | count | images |
|---|---|---|
| puncture / hole just above the grey band | 11 | 0160 0248 0520 0613 0687 0736 0832 0878 0955 0965 1071 |
| dent mid-body + scratch above the band | 10 | 0215 0402 0506 0623 0763 0781 0843 0904 1158 1170 |
| rust streak from the nose down | 5 | 0466 0471 0800 0935 1195 |
| best remaining residual score (no visible defect) | 2 | 0509 0896 |

The three types are isolated clusters (nothing in the 0.4-0.6 similarity band
around them), so the 26 look solid. Gate is 70 % (20 of 28).

Data note: the folder has 1,241 files; 16 `image_00xx(1).png` are byte-identical
copies of `image_0001..0016.png` and are skipped (1,225 unique images, not
1,125 as the statement says).

## Files to submit

- `test-output.txt`

## Method

1. `prep.py`: green-hue mask of the shell (largest blobs + vertical closing to
   bridge the grey band), principal axis, robust width from the cylindrical
   part, warp into an upright 80x256 frame (nose at top).
2. `score_pca.py`: Lab, per-image mean/std normalisation over the shell
   interior, remove the first 8 principal components (lighting, pose),
   blurred residual magnitude, score = peak. Nose tip excluded (crop jitter).
3. `detect.py`: residual maps of the top 40 are grouped by cosine similarity;
   groups that are isolated from the rest of the population are defect
   modes; each mode's mean residual is matched against all images; the list
   is filled to 28 by peak score.

```
python prep.py ../task_laos/images prep.npz
python detect.py prep.npz test-output.txt
```

No pretrained weights, CPU only, about 2 minutes.

## Risks

- Output format guessed (no `train-output.txt` in the data): one file name
  per line, with `.png`. Check against the reference file if the platform
  shows one.
- 2 of 28 are filler; there may be a fourth defect type that this misses.
