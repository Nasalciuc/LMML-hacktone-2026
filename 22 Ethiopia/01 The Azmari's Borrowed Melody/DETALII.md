# The Azmari's Borrowed Melody

A few seconds of a recording were replaced by something that wasn't played in that take. The task is to give the start and the end of that part, in seconds. The checker releases the flag when the answer is within 1.5 s on its hidden recordings (three tests on the platform).

**Status:** this `main.py` has not been submitted yet. Our first detector (ec2b674) failed test-3 on the platform and the second (b061ff3) failed test-2. Every number below comes from recordings we made ourselves, so none of it says whether the hidden tests pass.

## Run

    python main.py test-1-input.wav

Reads the WAV named on the command line from the current folder and writes `detection.csv` next to it (`start_s,end_s` and one row). Needs numpy and scipy only. About 2 s for a 56 s file.

On the train recording it answers `14.965, 20.912`. The truth is `14.960, 20.965`.

## Where this version comes from

`main.py` is the detector from `lmml-ethiopia-azmari-borrowed-melody.zip`, kept as received in `arhiva/`, with two changes:

1. **The melody features are left out of the search**: `f0log`, `centroid_f0`, `flux` and `mfcc1`–`mfcc6`. They follow which notes are played, not how or where the sound was recorded. On other melodies they moved the edges or pointed at an ordinary passage of the host.
2. **The block searched for is at most 20 s long**, down from 40 s. The task says "a few seconds", and in the tests every block longer than 20 s was a wrong pick.

The rest is the same code. `diff arhiva/main.py main.py` shows only these two changes.

## How it works

1. Every 50 ms it measures 37 features and uses 28 of them: harmonic ratios and their slope, tuning offset, energy below the fundamental, level at 20–120 Hz and below 12 Hz, DC drift, noise between the harmonics, noise floor, high-frequency share, 50 and 60 Hz hum, skewness and crest of the waveform, fast envelope modulation, envelope dips, pitch jitter.
2. Each feature is median-filtered over 5 frames, replaced by its rank and mapped onto a normal curve. The set is then whitened.
3. Every contiguous block of 1.5 s to 20 s (at most half the file) is scored by how far its mean lies from the mean of the rest. The best block is the answer.

No feature is assumed to move in a particular direction, so it doesn't matter which instrument is the host. If the search fails for any reason, a simpler low-frequency search answers instead, so a CSV is always written.

What gives the replaced part away in the train file is described in `arhiva/README.md`, with pictures in `arhiva/figures/`: another instrument (a washint inside a masenqo take), a higher register, tones at 30–95 Hz under each of its notes, and no drift below 10 Hz.

## Results

344 recordings with a known replaced part: 32 made from the train file and 312 synthetic ones (masenqo and washint, from the archive's generator). A recording passes when both edges are within 1.5 s of the truth.

| | Recordings | Archive | This version |
|---|---|---|---|
| All | 344 | 327 | 334 |
| Another instrument | 104 | 103 | 104 |
| Same instrument, another recording chain | 104 | 98 | 103 |
| Same instrument, same chain | 104 | 97 | 98 |
| Made from the train file | 32 | 29 | 29 |
| Replaced part under 2.5 s | 30 | 22 | 25 |
| Replaced part of 10–20 s | 54 | 53 | 53 |
| At the very start or end of the file | 48 | 48 | 48 |

The two changes fixed 7 recordings and broke none. Among the passed recordings, the larger of the two edge errors is 0.04 s at the median and 0.14 s at the 95th percentile.

What each change does on its own, and why the limit is 20 s and not lower (`check/ablation.py`):

| Variant | First 140 | Fresh 2.5–10 s | Short 1.5–2.5 s | Long 10–14 s | Long 14–20 s | At start | At end | All 344 |
|---|---|---|---|---|---|---|---|---|
| Recordings | 140 | 72 | 30 | 30 | 24 | 24 | 24 | 344 |
| Archive as is (max 40 s) | 134 | 70 | 22 | 29 | 24 | 24 | 24 | 327 |
| Max 20 s | 135 | 71 | 22 | 29 | 24 | 24 | 24 | 329 |
| Max 15 s | 135 | 71 | 23 | 29 | 9 | 24 | 24 | 315 |
| Max 12 s | 136 | 71 | 23 | 24 | 0 | 24 | 24 | 302 |
| No melody features | 135 | 71 | 25 | 29 | 24 | 24 | 24 | 332 |
| **No melody + max 20 s (this version)** | 136 | 72 | 25 | 29 | 24 | 24 | 24 | 334 |
| No melody + max 15 s | 136 | 72 | 26 | 29 | 9 | 24 | 24 | 320 |
| No melody + max 12 s | 137 | 72 | 27 | 25 | 0 | 24 | 24 | 309 |
| Chain features only | 132 | 67 | 21 | 29 | 22 | 24 | 22 | 317 |

Dropping the melody features is where most of the gain is (327 to 332). A limit under 20 s helps a few short parts but loses the long ones, so 20 s is the lowest safe value. Keeping only the recording-chain features is worse than the archive.

File formats (`check/formats.py`): the train recording saved as 8, 16 and 32-bit integers, float32, float64, stereo, 8, 16 and 44.1 kHz, ten times louder, ten times quieter, with a DC offset, with inverted polarity and cut short gives the right answer every time. Silence, a 0.2 s clip and a file that isn't audio all exit with 0 and a valid CSV, and nothing else is written.

## Not solved

- **Very short replaced parts (under 2.5 s):** 25 of 30. In the 5 misses the answer is somewhere else in the file.
- **Same instrument through the same recording chain:** 98 of 104. Only the playing differs, so there is little to go on. This is the likeliest way for a hidden test to fail.
- **Two unusual regions in one file:** the stronger one wins. On the three "F" recordings (a weak rumble splice plus a passage shifted an octave up) it gets 0 of 3, before and after the changes. That decoy does look foreign itself, so the test is harsh.
- **A replaced part longer than 20 s** can no longer be found.
- **No confidence value:** it always answers with a block. On the control recording with nothing replaced it answers 13.17–17.86.

Short parts and the same-instrument-same-chain case are where more work would pay off. Nothing was tuned for them here, because any tuning would fit the synthetic generator rather than the organisers' recordings.

## Folder

- `main.py` is the file to submit.
- `check/` has the scripts behind the numbers above, and `check/results.txt` has their output.
- `arhiva/` is the archive as received: its `main.py` (the baseline in the tables), its README, analysis and validation scripts and figures. Only `main.py` and `validation/synth.py` were run here. The other scripts use `/home/claude/...` paths and were not re-run.
- `varianta_veche/` is our earlier two-pass detector (ba2fb1b) with its README and `validate.py`. It was never submitted. On 36 synthetic recordings it passed 32, against 34 for our first detector, so it was set aside.

## Repeating the checks

    cd check
    python build_cases.py ../../task_ethiopia/train-input.wav <cache folder>
    python compare.py <cache folder>
    python ablation.py <cache folder>
    python formats.py ../../task_ethiopia/train-input.wav <empty work folder>

`build_cases.py` makes the 345 recordings and stores their features: about 5 minutes with 3 processes and about 115 MB, so keep the cache folder outside the repo. `compare.py` calls `detect()` from both files on every recording. `ablation.py` shows what each candidate change does on its own. `formats.py` runs `main.py` the way the grader does, and wipes its work folder first.
