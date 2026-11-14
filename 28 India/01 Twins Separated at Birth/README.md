# India - Twins Separated at Birth

## Results

| setup | F1 | gate |
|---|---|---|
| 5-fold CV on train (blocked by image index), content only | 0.976-0.978 | 0.88 |
| same, with the file-order cue (see below) | 0.996 | 0.88 |
| `main.py` on a 1,014-image train slice, zipped, 4 threads | 1.000 (343/343, in-sample) | 0.88 |

Runtime on that slice with 4 threads: 48 s wall (features 19 s,
verification 24 s), peak RSS 430 MB. First version was killed by the
grader after 143 s of feature extraction; v2 decodes JPEGs at reduced size
(DCT scaling), SIFT at 512 px / 400 keypoints, float16 descriptors, smaller
vocabulary, OpenCV internal threads off.

## Files to submit

- `main.py` (the tree model is embedded in the file)

Needs numpy + opencv-python (SIFT is in the main package since 4.4).
`DUP_THREADS=n` overrides the thread count (default: all cores).

## How it works

1. RootSIFT, 400 keypoints per image at 512 px (JPEG decoded at reduced size).
2. Candidates: bag of visual words built on the input itself (768 words,
   tf-idf, cosine), 10 nearest neighbours per image.
   Neighbours in file-name order (+1, +2) are always added too.
3. Each pair: mutual NN + ratio test, RANSAC similarity transform, then the
   aligned overlap is compared (intensity NCC and Laplacian NCC at 256 px).
   9 features: inliers, inlier ratios, scale, angle, reprojection error,
   NCC, Laplacian NCC, overlap.
4. Gradient-boosted trees (150 trees, depth 4) fitted on the train pairs and
   embedded as plain lists; evaluated with numpy.
5. Groups have max 3 images, so open triangles are closed.
6. File-order cue: if 50 %+ of the confident pairs are within 2 positions in
   sorted name order (train: 97-99 %), neighbours get a lower threshold and
   far-apart pairs are dropped (>90 %) or need p >= 0.95. On a shuffled set
   the cue switches itself off (checked on a shuffled copy of train: F1 0.976).

What the edits are in train: crops, resize, small rotations (< 4 deg, black
corners), colour / brightness shifts, grey-scale, blur. No mirror flips seen.
Most false positives on the content-only path are real web re-posts of the
same photo (watermarked, re-cropped) that are not in the label file.

Input: `python main.py <zip or folder>`; without an argument it takes the
first `.zip` in the cwd (prefers names with input/test), else a folder
`input`, `test`, `train`, ... Output always goes to `./output.csv`.

## Risks

- Runtime limit is not stated; ~1 min per 1,000 images on 4 threads here.
- The file-order cue fits how train was built. If the hidden set is shuffled
  the code falls back to content only (~0.976 on train CV).
