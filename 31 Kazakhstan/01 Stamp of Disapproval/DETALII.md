# Kazakhstan: Stamp of Disapproval

**Task.** 5 albums x 26 pages, 3 stamps per page, exactly one stamp per page altered in one small region (digit changed / element cloned / inscription removed). Marking the altered region of every page of an album on one sheet draws a letter; five letters give `SIGMOID_<WORD>`. 30 altered single stamps with masks (exhibits) are the only ground truth.

**Flag: not found.** The content-reading detector (section 1) pins the altered stamp on 42 of 130 pages with high confidence and on about 32 more with medium confidence, but the dots of no album read as a letter, and a template fit of A-Z is no better than random points. No word guess is offered.

## Data findings

- 130 pages, 394 stamp crops (127 pages give 3 crops, 3 pages are over-segmented into 4-5).
- Exhibit edit types, from the masks: 10 digit changes (ex 2, 7, 9, 12, 13, 17, 22, 24, 25, 26), 2 clones (15, 19), 18 removals of a small inscription (mostly the year of issue or the designer's name near an edge).
- Digit changes produce values that do not exist: 55, 66, 99, 133, 155, 110, year 2211. On album pages the same pattern appears: 202 (x6), 110, 33, 550, 155, 66, 88, 22, 101, 55, 188, 244, 252, 303, 212, 232, 44, 99.
- Removals: constant colour, constant colour + white noise (std 2-3), or smooth fill. Swapped digits are re-rendered, not pixel copies.
- Designs are not reused: 404 of 417 occur once, so copy comparison is impossible (and excluded by the rules).

## 1. Content reading with a VLM (folder `vlm/`, latest attempt)

Model: Qwen2.5-VL-7B-Instruct, 4-bit (about 6 GB VRAM), about 5 s per stamp batched.

1. `crops.py` cuts each page into stamps (`crops.json` keeps the page box of every crop).
2. `vlm_fast.py` reads every stamp into JSON: country names, face values, years, other text (`vlm_albums.json`). `vlm_read.py` is the first, unbatched version used for the exhibits; it also asks "was it altered?" (`p_alt`) and "where?" (`box`).
3. `content_rules.py` turns a read into cues. **Strong**: a face value with a repeated or copied digit (55, 155, 202, 133...) or a year after 2026. Weak: a value not divisible by 5. Values with a leading zero are treated as misreads.
4. `ground.py` asks the VLM for the box of the odd number (ordinary grounding), giving the point on the page.
5. `yearq.py` asks "is the year of issue printed?" and keeps P(No) per stamp.
6. `decide.py --mode tiered` per page: (a) the single stamp with a strong cue, dot at the grounded number; (b) else the stamp whose year is clearly missing (P >= 0.9 and 0.3 above the next stamp), dot at the forgery-net peak inside that stamp; (c) else the forgery-net peak (weak fallback).
7. `fit_letters.py` fits A-Z stroke templates to the confident dots and compares with random points.

### Exhibit accuracy (30 exhibits, all altered)

| check | result |
|---|---|
| strong content cue fires, digit changes | 8 / 10 (missed: ex07 "100", ex26 overlapped digit) |
| strong content cue fires, removals + clones | 0 / 20 (as expected, the value is untouched) |
| grounded number box overlaps the mask, on the 8 flagged | 8 / 8 (centre inside mask 6 / 8) |
| "year of issue missing" P >= 0.7, removals | 13 / 18 |
| same question on presumed genuine album stamps (false alarm) | 34% at 0.7 |
| VLM "was it altered?" above 0.5 | 2 / 30 (mean 0.31) - useless |
| VLM "where is the alteration?" box centre in mask | 4 / 30 - useless |
| "empty area near an edge?" question | no separation (median 0.59 everywhere) |

So in total the new detector localises 8 / 30 exhibits precisely (all digit edits) and flags the right kind of defect on 13 more removals without a position.

### Album pages

- Strong cue: exactly one stamp on 42 pages, two stamps on 0 pages, none on 88. With about a third of edits being digit changes, about 43 pages were expected, so these 42 are very likely correct picks.
- Cross-check on those 42 pages: the forgery net's top stamp agrees 18 times and `p_alt` 11 times (chance is 14). Both earlier signals are at chance for choosing the stamp, which explains why the earlier sheets were noise.
- Year-missing tier adds 32 pages; estimated about 70% correct, and the dot inside the stamp is only the net peak.
- Remaining 56 pages fall back to the net (near chance).

| album | strong cue | year missing | net fallback |
|---|---|---|---|
| 1 | 6 | 8 | 12 |
| 2 | 6 | 9 | 11 |
| 3 | 11 | 5 | 10 |
| 4 | 13 | 3 | 10 |
| 5 | 6 | 7 | 13 |

Sheets: `vlm/album_sheets_tiered.png` (black = strong, blue = year missing, grey = fallback) and `vlm/album_sheets_tiered_confident.png` (without the fallback dots). Boxes: `vlm/boxes_tiered.csv`.

### Reading the letters

Not possible. By eye none of the five sheets is a letter, including album 4 with 13 trusted dots. `fit_letters.py` best fits (lower is better, against the same number of random points): album 1 C 20.5 (random 21.9 +- 1.5), album 2 M 20.6 (23.3 +- 2.3), album 3 R 17.9 (18.7 +- 1.9), album 4 D 24.7 (25.6 +- 1.2), album 5 A 17.1 (15.7 +- 1.1). Nothing beats the null, so "CMRDA" is not a guess, just the argmax of noise.

What is still missing: a detector for the two thirds of pages with removals/clones that both chooses the stamp and gives the position. Half of a letter's dots do not show the letter.

### Follow-up checks (structure, layout, patch cue)

- **Coordinates.** All 130 pages are 1600x1200 and dots are raw page pixels, so the mapping is not the problem. `vlm/strong_points_raw.png` shows the 42 strong-cue points with the stamp boxes and page index.
- **Slot / page order.** On the 42 strong pages the altered stamp's left-to-right slot is spread evenly (album 4: 4 / 5 / 4) and the page index shows no stable relation to the dot position (correlations change sign between albums; n is 6-13). No ternary or alphabet-index code is visible; the statement also says file names carry no meaning.
- **Layout leak.** The altered stamp is the most central of the three on 26 / 42 strong pages, second on 14, least central on 2 (chance 14 / 14 / 14). Median distance of its centre from the page centre is 276 px against 442 px for the neighbours. The dots sit in the middle of the page because the letter does. This is a prior for choosing the stamp, not a detection.
- **Cross-stamp flat-patch cue** (`vlm/patch_cue.py`, `patch_feats.json`): flat 16x16 windows, residual std and lag-1 autocorrelation, robust z against the stamp's own flat windows. Exhibits: peak inside the mask on 8 / 30 (all removals, 8 / 18). Across stamps it is weak: its top stamp is the most central on 44 / 88 non-strong pages (33% would be chance, about 62% a correct detector).
- **Combined sheet** `vlm/album_sheets_bayes.png` (`decisions_bayes.json`, `boxes_bayes.csv`, built inline, not by a script): strong cue where present; otherwise centrality prior (26/14/2) x year-missing likelihood ratio (2.1 / 0.42) + 0.15 x patch z, dot at the patch-cue peak. Blue dots are therefore prior-driven and their position inside the stamp is right only about 4 times in 10 at best.
- **Reading.** Album 5 now resembles an "A" (apex at the top, two legs, bar), which is also the best template for album 5 in `fit_letters.py`; album 2 could be "N" or "V". Albums 1, 3, 4 are unreadable. Confidence in any letter is low and there is still no word.

### Reproduce

```
cd vlm
python crops.py
python vlm_read.py --what exhibits          # about 15 min
python vlm_fast.py --what albums --bs 6     # about 30 min
python net_feats.py                         # needs ../outputs_v1, ../outputs_v2
python ground.py --what exhibits && python ground.py --what albums
python yearq.py --what exhibits && python yearq.py --what albums
python decide.py --exhibits
python decide.py --mode tiered              # also: content, hybrid, net
python fit_letters.py
```
Python: `C:\Users\user\miniconda3\python.exe` (torch, transformers, bitsandbytes, qwen_vl_utils). Set `PYTHONIOENCODING=utf-8`; `HF_HUB_OFFLINE=1` once the model is cached.

### Files in `vlm/`

- `crops.py`, `crops.json`, `crops/`: stamp crops
- `vlm_read.py`, `vlm_fast.py`, `vlm_exhibits.json`, `vlm_albums.json`, `log_*.txt`: reads
- `content_rules.py`: cues from a read
- `ground.py`, `ground.json`, `ground_exhibits.json`: boxes of the odd numbers
- `yearq.py`, `yearq_*.json`: year-missing scores
- `net_feats.py`, `net_feats.json`: forgery-net peak per stamp
- `decide.py`, `decisions_<mode>.json`, `boxes_<mode>.csv`, `album_sheets_<mode>.png`: decisions and sheets
- `fit_letters.py`: letter template fit

---

# Earlier attempts (pixel forensics)

None of these sheets reads as a letter either.

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
