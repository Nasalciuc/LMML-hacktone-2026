# City of Subtitles: camera attribution

Task (LMML 2026, Japan): 145 nameless photos (weeks 8-12) from six regulars. Attribute them from sensor noise
(PRNU) using the 108 credited photos of weeks 1-7, find Toorisugari's, and recover the hidden word for
`SIGMOID_<WORD>`. Week-13 hint: "The photos of a simple passer-by might know more than they show. Who knows,
maybe an unknown photographer identity hides there."

## Results

- All 145 anonymous photos attributed: `attribution_table.csv` (file, photographer, rotated180, z, margin).
- Counts: toorisugari 45 (exactly 9 every week), every other regular 20 (4 every week). 29 = 9 + 5 x 4 each week.
- Toorisugari's photos: w08_002 004 006 010 011 013 017 019 023 · w09_001 004 008 011 013 014 015 021 022 ·
  w10_010 011 012 018 019 020 022 027 028 · w11_003 011 012 014 016 025 027 028 029 ·
  w12_001 002 005 008 010 020 025 027 029.
- Leave-one-out on the 108 credited photos: 105/108 correct (all 18 toorisugari, all 18 yakin correct;
  the three misses are amayadori, the weakest sensor).
- Median z of the chosen camera 16.7; 133/145 photos have a margin > 5 over the next camera; 43/45
  toorisugari photos have margin > 5.
- Flag: **not recovered** (see "The secret" for everything ruled out, and the open leads at the end).

## Method

1. Residual (`feats.py`, `wavelet.py`): per colour channel, image minus a db4 wavelet Wiener denoise
   (Mihcak style, sigma 3, 4 levels). Divided by its local RMS (15x15) so scene edges do not dominate,
   row and column means removed. Caption bands (black rows) and clipped pixels masked.
2. Fingerprint: MLE estimate K = sum(W·I) / sum(I²) per photographer; the mean of the six fingerprints is
   subtracted (common processing traces).
3. Score: normalised correlation of the photo residual with K·I, turned into a z-score with the spread of the
   same correlation at 12 large circular shifts (null).
4. Refinement: anonymous photos with margin > 3 are added to their fingerprint, repeated until stable
   (leave-one-out always). Then a weekly quota (9 + 4 x 5) fixes two weak photos (`finalize_table.py`).

Leave-one-out accuracy on the credited photos by step: scipy Wiener residual + NCC 56%, local Wiener +
whitening + common-part removal 72%, wavelet residual 79%, + iterative refinement 86%, + orientation
(below) 97%.

### The twist that matters: photos turned upside down

The first attribution split toorisugari into two groups that do not correlate with each other at all
(fingerprint correlation 0.0005), as if a seventh, unknown camera were hiding in its photos. It is the same
sensor turned 180 degrees: the fingerprint of the "unknown" group against the toorisugari fingerprint
rotated by 180 degrees gives PCE 1542 at zero shift (`orient_check.py`). Some photos were shot with the
phone upside down and posted upright.

This happens to every regular, not only toorisugari (rotated share 22-56 % per person), so the final
attribution (`attribute_final.py`) scores every photo against every fingerprint in both orientations and
accumulates fingerprints in sensor orientation. That is what lifts leave-one-out from 86 % to 97 % and makes
the weekly counts come out exact. Mirroring left-right / top-bottom was tested too (`orient4.py`), no photo
needs it.

Yakin vs toorisugari: their sensor fingerprints correlate at 0.0018 (same level as any other pair), so the
"same phone model" adds nothing a PRNU test cannot separate; at 97 % no yakin/toorisugari confusion is left.

## The secret: what was checked (all negative so far)

- Orientation bits of toorisugari's photos (posting order, per week, as 5x9 grid, 5/7/8-bit readings,
  inverted): no text; other regulars rotate at the same random-looking rate.
- Positions of toorisugari's photos in the week x counter grid (5x29 bitmap, 5/7/8-bit readings): no text.
- A second pattern in toorisugari's photos: posted frame (`posted_pattern.py`, `posted_additive.py`,
  `posted_render.py`), sensor frame anonymous vs credited (`extra_component.py`), pairwise after removing its
  own pattern (`pairs_after_removal.py`), block-wise presence of its pattern per photo (`photo_presence.py`).
- Pictures inside fingerprints: smoothed, local energy, texture bands, colour differences, row/column
  profiles, local agreement with every other camera (`render_sensor.py`, `texture_maps.py`,
  `sensor_xcorr.py`, `line_patterns.py`).
- PRNU gain per photo (`prnu_gain.py`), shifts off the sensor grid (`shift_scan.py`).
- Pixels/file: no metadata, identical quantisation tables and marker layout, nothing after EOI, no hidden
  rows past the header height, black bands are pure 0, DCT coefficient histograms show no JSteg-style
  pair equalisation (`dct_stats.py`).
- Captions, now OCR'd for all 253 photos (`caption_ocr.py` -> `out/captions_ocr.json`, manga-ocr):
  every caption is 1-6 lines (not the "one line" of the story), built from random "fragment, fragment"
  combinations (about 490 first halves and 310 second halves after OCR noise). Toorisugari's line counts
  (1-6) are spread like everyone else's (`out/caption_linecounts.csv`). In posting order, his first characters
  are 秋遠最雨雨開冬雨空深誰夜夜誰息鏡遠答..., which is no acrostic. Of the 87 fragments read the same way in at least 4 photos,
  none appears only in his 63 photos (random 63-photo groups: 0.13 on average). Listing in `out/captions_toorisugari.txt`. Conclusion: the captions are decoration.
- Orientation bits read across weeks (9 columns of 5 bits, `rotation_columns.py`): best is `WET?MXDLT`;
  3 letters forming a word by chance is likely, and column 4 is 28 (no letter). Not it.
- Attributed-photographer sequence per week as initials (Y A S J R T) or as indices 0-5, and the gaps between
  Toorisugari's counters per week: nothing readable.
- Posted-orientation mean residual of his 45 anonymous photos, his own sensor term regressed out, rendered at
  full resolution (Gaussian 1.5 and 4 px, gradient, 9 px energy) instead of block means, which cancel letter
  edges (`posted_fullres.py`, `out/pf_*.png`): only scene leakage and more noise at the bottom, where captions
  mask fewer photos. No text.
- `out/hidden_pattern_z.csv` cannot be used as a per-photo bit: all toorisugari z are around -100 because
  the photo is part of P's construction (bias), not because a pattern is present or absent.

### Hidden seventh sensor: pairwise and pooled search (all photographers)

Goal: find photos (anyone's) that share a sensor pattern other than their own, possibly shifted, cropped,
mirrored, turned or rescaled. If a group showed up, its fingerprint would be estimated and rendered, and its
membership read as bits. No group showed up, so nothing was rendered or read.

1. `clean_residuals.py`: luma residual of every photo minus its own photographer's leave-one-out sensor term
   (fitted per photo, turned to the photo's orientation) -> `cache/clean_res.pt`.
2. `pce_pairs.py`: all 31,878 pairs, full circular shift search by FFT, relative transforms id, rot180,
   mirror LR, mirror UD (960x1280) and rot90, rot270 (960x960 centre crop). Crops are covered by the shift
   search. Plain correlation gave 1097 hits above PCE 60 (`out/pce_report.txt`), but the median PCE is 40
   (white noise gives about 28), so the residuals are not white. `pce_pairs.py phat` (phase-only correlation)
   brought it down to 47 hits, median 31 (`out/pce_report_phat.txt`).
3. The surviving top hits all involve five very dark night photos (amayadori w06_009, jihanki w06_006,
   toorisugari w02_013, w02_018, w05_013; `out/pce_hubs.png`). They match each other and many others in
   every transform at different shifts, and 12 of their 34 hits sit at shifts that are multiples of 8 in both
   directions (chance: 1 in 64). That is JPEG block structure in flat dark areas, not a sensor: a shared
   sensor would match in one geometry only.
4. Without the hubs, the tail is what noise gives: 13 pairs above 60 against 11.6 expected from an exponential
   fit of the tail, 1 above 70 against 0.5 expected. The largest connected groups (13, 11, 7 photos) are
   chains (edges = members - 1), with mixed transforms and photographers.
5. `pce_scales.py`: toorisugari's 63 photos against all 253 with the second photo rescaled by 0.90 / 0.95 /
   1.05 / 1.10 (id and rot180, PHAT): the only hits above 60 are the same dark hubs (`out/pce_scales.log`).
6. Power check (`pce_power.py`): the same pairwise test on uncleaned residuals of photos that really share a
   sensor gives PCE 25-43 and never peaks at zero shift, the same as pairs from different sensors. So one photo
   per side carries too little PRNU here, and the pairwise search can only catch a hidden sensor much stronger
   than the real ones.
7. Pooled tests, which do have power (each photo against the mean of many others, z against 16 shifts).
   - `loo_shared.py` (cleaned residuals, leave-one-out): every z is strongly negative. Each photo was cleaned
     with a fingerprint that contains the others, which builds in a negative bias. This is also why the old
     `hidden_pattern_z.csv` is unusable.
   - `loo_shared2.py` (split halves, each half cleaned with the other half's fingerprint): every z is positive,
     highest in sensor frame. A fingerprint from about 19 photos is too noisy to remove the own pattern, and
     the leftover dominates.
   - `loo_shared3.py` (raw residuals, no cleaning needed):
     T1, a posted-frame pattern within a photographer: each photo against the mean of the same photographer's
     photos with the opposite rotation flag. Their sensor patterns are 180 degrees apart there and do not
     correlate. Mean z per photographer is -0.4 to 1.2 with sd 1.6-1.9; toorisugari's is 0.61, the same as
     yakin's 0.59. No group has a tail.
     T2, a pattern shared across photographers: each photo against the mean of all other photographers, posted
     and sensor frame. Means -0.5 to 1.0, max z 5.1 (one rojiura photo), no group tail.
     Toorisugari's T1 z in posting order look like noise (one photo at -7.3), so there is no membership pattern
     to read.

Conclusion: no second sensor is detectable in anyone's photos, in any of these geometries. The ~3 sigma
"posted pattern" seen earlier does not hold up once the bias is removed.

### Credited vs anonymous fingerprint (`k_split.py`)

If the account owner had added an "unknown identity" to his anonymous photos only, any fingerprint built from all
his photos would have absorbed it, and cleaning with that fingerprint would have subtracted it. So his fingerprint
was estimated twice, separately: K_c from the 18 credited photos, K_a from the 45 anonymous ones (luma, MLE,
each photo turned to the sensor frame).
- Sensor frame: D = K_a - a*K_c. Posted frame (K_a not turned): D = K_a - a0*K_c - a1*rot180(K_c).
- D is multiplied by sqrt(sum I^2) so its noise is flat. Without that, the rows near the captions, covered by
  fewer photos, light up and mimic a shape.
- Rendered as raw, Gaussian 2/4/8, |D| blurred and local energy above its 90th percentile
  (`out/ksplit_<who>_<scope>_<frame>.png`, plus `_gauss8_full` and `_energy16_full`). Also done per week (9 photos
  each, `out/ksplit_weeks_montage.png`) and for yakin (credited 18 vs anonymous 20) as the control.
- corr(K_a, K_c) is 0.039 (yakin 0.028); a is about 0.02 because K_c from 18 photos is mostly estimation noise.
- Result: the energy and |D| maps are uniform speckle in both frames, for all scopes. The blurred maps show
  only ghosts of the scenes: building facades and branches, and in week 10 a large round shape where that
  week's 9 scenes put it. No letters or shapes, and toorisugari looks like the yakin control.

## Fresh look (14 Nov, second pass)

Four new directions, each stopped as soon as it was clearly negative. Scripts and CSVs are in `fresh_look/`.
The word was **not found**. Short version: his 45 photos look like one homogeneous set in every measurable way,
there is no "4 normal + 5 extra" split, and nothing is hidden in the JPEG coefficients, the scene, or the grain.

### 1. The 25 extra photos (9 per week instead of 4)

Looked for anything that splits his 9 weekly photos into 4 + 5.
- Processing features per photo (`feat1.py`, `feat2.py`, `warm.py`): file size, mean colour and R-B balance,
  brightness, noise per channel, grain autocorrelation, grain colour correlation, caption band height/position,
  line count, text pixel count, caption text colour, highlight and shadow colour cast. A two-cluster gap score per
  feature is in the same range for him as for the other five (e.g. size 3.3 vs 2.8-5.0, R-B 3.7 vs 2.6-4.4).
  The only feature where he stands out is noise level (6.2 vs 2.6-4.2), and that is three noisy night shots
  (w10_027, w12_002, w12_005), not five per week.
- JPEG container (`markers.py`): all 253 files have the same marker layout, the same two quantisation tables
  (sums 1109 / 1666), 4:2:0, and the same Huffman tables. Nothing per photo.
- Fingerprint strength: PRNU gain per week sorted (0.073-0.112 in w08, 0.053-0.156 in w10 ...) is a smooth
  spread, no step. Credited and anonymous gains match.
- A second camera inside his photos: second-best z of his 45 photos is 2.75 on average (others 2.6-3.4), max 6.8.
- Hidden shared pattern in a subset (`gram2.py`, `gram3.py`): 253x253 correlation matrix of the cleaned residuals,
  in posted, sensor and anti-sensor geometry. For his 45 anonymous photos the largest eigenvalue is 12.4 / 13.3 /
  12.0 against a pure-noise edge of 13.4; for all 63 it is 14.3 / 16.0 / 14.4 against 15.9. The other five give
  the same picture. A subset of 25 sharing a pattern of sensor strength would give about 25. A +/- sign code
  would show the same way, and does not.
- Pooled presence of his own pattern (`presence_pool.py`): the per-pixel product residual x expected pattern,
  summed over the 45 photos, 16 and 32 px blocks, posted and sensor frame. Flat speckle, no region where his
  pattern is missing or doubled; same for the other five.
- Low-frequency mark (`lowmean.py`): the earlier tests all used a denoising residual, which removes anything
  smooth. Here the raw image minus a 8 px / 32 px blur was averaged over his 45 photos. Only overlapping scene
  edges; amplitude 3.3 against 2.0 for the other 100 photos, which is the 1/sqrt(N) ratio (expected 1.5, got 1.6).
- Scene duplicates (`dup.py`): none across the 253 photos, also not turned or mirrored.

One oddity, not in his photos: `dualscan.py` scores every photo against fingerprints built separately from
credited and from anonymous photos. The credited amayadori photo w02_015 carries amayadori's pattern (z 11.2) and
also toorisugari's, turned 180 degrees, in both independent fingerprints (z 6.8 and 9.1; null sd is 2.4, so about
2.8 and 3.7 sigma). It is the only photo like that (next best pair is 5.4 / 7.8), so it is a single unexplained
outlier and not a code.

Conclusion: no extras. The 9 per week are one set.

### 2. Scene content

Qwen2.5-VL was not run: the shared GPU had 3.5-5.6 GB free and the 4-bit 7B model needs more. Instead all 45 photos
were read by eye at 640x480 (`tw.py` makes the weekly 3x3 sheets), plus a dark-range and a bright-range contrast
stretch. They are ordinary street photos from different countries with no weekly theme. The text in them is
normal signage (POKKA, Takakura-dori St., KOREAN BBQ, JOB FAIR, MUSEUM OF SCIENCE, Coca-Cola, AQUARIUS) and the
other regulars' photos have the same kind. No drawn letter, number or shape, nothing in the shadows or highlights.
Negative, with the caveat that a VLM pass on small signs was not done.

### 3. JPEG steganography

Real quantised coefficients read with jpeglib (`dct1.py`, `lsb.py`), all 253 photos.
- Pair statistics: count(2)/count(3) is 1.83 for him against 1.83-1.86 for the others, where JSteg or OutGuess
  would push it towards 1. count(2)/count(1) is 0.350 against 0.348-0.351, so no F5 shrinkage. LSB share of
  non-zero AC coefficients 0.687 against 0.687-0.692. Credited and anonymous agree.
- LSB streams: luma and both chroma planes, zigzag and natural order, JSteg rule / all non-zero / all AC / DC only,
  plain and inverted, both bit orders, 8 bit offsets, first 4000 bytes. Printable runs of 10+ characters: 12.5 per
  photo for him, 12-14 for the others, all gibberish. No SIGMOID, no FLAG.

Negative.

### 4. Grain and processing parameters

- The grain is the same in R, G and B (channel correlation 0.996 in flat areas), sd about 7-9 grey levels, slightly
  blurred (neighbour correlation 0.38). Same for everyone.
- Local grain strength maps of his 45 photos (`grain.py`): scene edges only, flat sky is uniform. No letters.
- Grain strength per photo: his median 8.99, the others 7.4-8.7, his credited photos 8.84. It follows the scene.
- Warmth: highlights are shifted about +20 R-B, shadows have blue crushed to 0, for everyone. Per photo it follows
  the scene and has no two-level structure. Caption text is neutral grey (231-250), so captions are drawn after
  the colour step; the grey level varies the same way for all six.

Negative.

### Where this leaves it

Ruled out on top of the earlier list: an extra subset among his photos, a second camera in a subset (any of three
geometries, including a sign code), a smooth watermark, coefficient-domain stego, planted scene content, grain
shape, grain strength, warmth. Still untested: a VLM read of small signs, and a hidden pattern placed at a
different shift in every photo (the pairwise shift search has no power for that).

## Open leads

None strong. Untested only: a hidden pattern that is not sensor-like (different spectrum or very low
frequency) and lives in a geometry other than posted or sensor frame; and scales outside 0.90-1.10.

## Files

| file | role |
|---|---|
| `prnu.py`, `wavelet.py`, `feats.py` | loading, caption mask, residual extraction (cache in `cache/`, not needed in the repo) |
| `attribute_final.py` | six cameras x two orientations, iterative attribution -> `attribution.csv`, `credited_check.csv` |
| `finalize_table.py` | weekly quota -> `attribution_table.csv` (the answer table) |
| `attribute2.py`, `attribute7.py`, `groups.py`, `orient_check.py` | earlier stages that led to the orientation finding |
| `caption_ocr.py`, `caption_report.py` | OCR of all captions, toorisugari's in posting order |
| `rotation_columns.py`, `posted_fullres.py` | orientation bits as letters; full-res posted-pattern renders |
| `clean_residuals.py`, `pce_pairs.py` (+`pce_pairs_lib.py`), `pce_report.py`, `pce_scales.py`, `pce_power.py` | hidden-sensor pairwise PCE search (steps 1-6) |
| `loo_shared.py`, `loo_shared2.py`, `loo_shared3.py` | pooled hidden-pattern tests; only `loo_shared3.py` is unbiased |
| `k_split.py` | fingerprint from credited vs anonymous photos, difference rendered (all, per week, yakin control) |
| other `*.py` | the checks listed under "The secret" |

Run: `python feats.py && python attribute_final.py && python finalize_table.py` (GPU, about 15 min).
Hidden-sensor search: `python clean_residuals.py && python pce_pairs.py phat && python pce_report.py 60 pce_pairs_phat.npz`,
`python pce_scales.py`, `python loo_shared3.py` (all with `-X utf8` on Windows; about 1 GB VRAM).
Dataset expected at `../task_japan/dataset`.
