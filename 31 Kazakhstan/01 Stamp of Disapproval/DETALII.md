# Kazakhstan: Stamp of Disapproval

**Flag: not found yet.** None of the five album sheets reads as a letter. Below is what was built, what it finds and why it is not enough.

## Pipeline

1. **Stamps.** Each page is split into its stamps: everything that differs from the dark album background (`stamps_on`).
2. **Forgery network** (`forgery_net.py`). A U-Net trained from scratch that outputs a per-pixel "altered" map.
   - Inputs: RGB plus a 3x3 high-pass residual.
   - Training data, part 1: synthetic edits on the album stamps, mirroring the three edit types in the brief:
     - digit change: a digit-sized patch copied sideways, either whole or strokes only;
     - element clone: a soft blob copied elsewhere;
     - inscription removal: a plain-background rectangle filled with its border colour, either with or without fresh white noise, or inpainted.
   - Edits are made at page resolution (sometimes at a higher resolution, then resampled), then the stamp is re-saved as JPEG q90-98.
   - Training data, part 2: the 30 exhibits with their real masks, used for 25-40% of the batches.
3. **Pages** (`detect_pages.py`, `sheets.py`).
   - The network runs on every stamp, and the strongest smoothed peak on each page gives that page's region.
   - Outputs: `boxes.csv`, the per-page overlays in `pages/`, and two kinds of album sheet. `album_sheets_top1.png` shows one dot per page. `album_sheets_overlay.png` sums the 26 normalised heatmaps of each album.

## Results

- **Held-out exhibits** (8 exhibits; the model trained on the other 22): 2-4 of 8 regions found, depending on the training step (`train_log_val.txt`).
  - It finds removals on plain paper well.
  - It also fires on genuinely plain corners.
  - It mostly misses digit swaps.
- **Album pages:** `outputs_v1` comes from the held-out model, `outputs_v2` from the model trained on all 30 exhibits.
  - The two models agree on the page region for 8-13 pages per album.
  - Neither sheet shows a clean letter. Album 5 has a dense cluster at the upper left, but it is not readable.

## Classical cues tested on the exhibits

Each cue is scored by how often its strongest point falls inside the exhibit mask (30 exhibits):

| cue | file | hits / 30 |
|---|---|---|
| copy-move: 12x12 residual block matching with displacement coherence | `cues.py` | 1 |
| same, with edges removed (grain only) | `cues.py` | 0 |
| flatness: residual energy below the stamp's grain | `cues.py` | 4 |
| white-noise fill: residual lag-1 autocorrelation near 0 on flat areas | in notes below | 8 |
| noise level against brightness-conditioned expectation | `noise_cue.py` | 7 |
| old JPEG block grid (period 8 x scale) out of phase | dropped | 5 |
| chroma high-frequency excess | dropped | 2 |

What the exhibits show:
- Removals come in three kinds: constant colour, constant colour plus spatially white noise (std 2-3, autocorrelation about -0.13 against +0.15...0.3 for real print), or a smooth fill.
- Swapped digits are not pixel copies. The grain does not match the source digit, so the copy was resampled or re-rendered.

The white-noise cue explains 5 of the 9 removal exhibits. On the album pages, the same cue gives scattered points.

## Reference-copy check

The idea was to align each stamp to other copies of the same design and look at the residual. This does not work here.

- `copies.py` matched every pair of the 387 album stamps with ORB and a RANSAC homography, then correlated each aligned pair.
  - 368 pairs align on some shared feature, usually text or a border.
  - Only 6 pairs (12 stamps, at most 12 of 130 pages) are the same design with correlation above 0.9. Even these look like siblings from one series rather than identical stamps.
  - The other 375 stamps have no copy anywhere in the albums.
- Thumbnail clustering of album stamps plus exhibits gives the same picture: 404 of 417 designs occur once. One exhibit (exhibit_04) has a near match on a page.
- `copy_residual.py` scores the 12 stamps that have a copy (`copy_residual_log.txt`), but 12 pages are far too few to draw letters.
- This fits the brief: there are "no untouched copies of the altered stamps", and comparing against copies is excluded as detection.

## Files

- `forgery_net.py`: synthetic edits, model and training
- `detect_pages.py`, `sheets.py`: page inference, `boxes.csv`, sheets
- `cues.py`, `noise_cue.py`: classical cues
- `copies.py`, `copies.json`, `copy_residual.py`, `copy_residual_log.txt`, `copy_residual.json`: design matching and copy residuals
- `forgery_net.pt`: model trained on all exhibits; the final weights are the step-6500 checkpoint, because later steps hit a CUDA error
- `forgery_net_val8.pt`: model with 8 exhibits held out
- `outputs_v1/`, `outputs_v2/`: boxes, sheets, heatmaps, page overlays
- `train_log_val.txt`, `train_log_final.txt`: run outputs

The environment is a local venv with opencv-python-headless and scikit-image, plus torch. No pretrained weights are used.
