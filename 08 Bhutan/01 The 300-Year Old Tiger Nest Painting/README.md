# Bhutan - The 300-Year Old Tiger Nest Painting

## Results

| check | gate | local |
|---|---|---|
| train strips (192 x 8 px) order match | >= 0.90 | 1.000 (exact, not mirrored) |
| train SSIM | >= 0.90 | 1.000 (pixel-identical to train-output) |
| runtime on train | < 20 s | ~3-4 s |

Extra checks with images shredded locally (same 1536x1024, 8 px strips):
train image flipped / cropped / 4 px and 6 px strips: all exact.
Other photos from the repo (India, Mongolia, Japan): exact on all textured
images. Failures only on images with large flat margins (album pages,
black-background rubbings), where several strips are identical and the
start of the image is ambiguous. Not the case for the painting.
Longest run: 9.4 s (384 strips of 4 px).

## Files to submit

- `main.py` - reads `input/`, writes `reconstructed_image.png` and `reconstructed_order.json` next to itself
- `reconstructed_image.png`, `reconstructed_order.json` - output on the train strips

Env: Python 3.13, numpy 2.5.3, opencv-python 5.0.0.93 (scipy / scikit-image not needed). Tested in a venv with the pinned versions.

## Method

1. Edge cost for "strip j right after strip i" from the two outer pixel
   columns of each side: linear prediction error from both sides
   (sqrt-norm of |actual - extrapolated|), plus a small Mahalanobis gradient
   compatibility term (MGC) as tie breaker. Fully vectorised.
2. Costs normalised by the second-best alternative (row and column) with a
   floor, so flat areas do not blow up.
3. Order = shortest Hamiltonian path on the asymmetric matrix: greedy edge
   merging without cycles, then segment moves (or-opt) until no gain.
   The cost is directional, so the path also fixes left/right orientation
   (important because SSIM is computed against the unmirrored image).

Run: `python main.py` (folder `input` in the cwd or next to the script), or `python main.py <folder>`.

## Risks

- If the hidden image has big uniform margins on both sides, the start
  point of the order can wrap around. The train painting has dark torn
  edges, so this is unlikely.
