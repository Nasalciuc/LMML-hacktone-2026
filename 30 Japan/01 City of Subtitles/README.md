# City of Subtitles: camera attribution

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
- Flag: **not recovered yet** (see "The secret" for what was ruled out).

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
- Captions: random combinations of the same phrase pool for everyone (read for week 8).

## Files

| file | role |
|---|---|
| `prnu.py`, `wavelet.py`, `feats.py` | loading, caption mask, residual extraction (cache in `cache/`, not needed in the repo) |
| `attribute_final.py` | six cameras x two orientations, iterative attribution -> `attribution.csv`, `credited_check.csv` |
| `finalize_table.py` | weekly quota -> `attribution_table.csv` (the answer table) |
| `attribute2.py`, `attribute7.py`, `groups.py`, `orient_check.py` | earlier stages that led to the orientation finding |
| other `*.py` | the checks listed under "The secret" |

Run: `python feats.py && python attribute_final.py && python finalize_table.py` (GPU, about 15 min).
Dataset expected at `../task_japan/dataset`.
