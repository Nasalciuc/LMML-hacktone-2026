# Ethiopia - The Azmari's Borrowed Melody (LMML, Medium)

Task: find the start and end (seconds) of a phrase that was spliced into a
recording. Submission is `main.py`; the checker releases the flag
(`SIGMOID_<WORD>`) only when all three hidden recordings are within 1.5 s.
The flag is not in the provided files, so it is not recorded here.

## Run

    python main.py test-1-input.wav     # writes detection.csv next to it

Needs only numpy and scipy. About 1.3 s for a 56 s file.

Train result: `14.965, 20.912` (truth `14.960, 20.965`).

## What gives the splice away (train file)

- Different instrument: the insert is a washint (odd harmonics dominate,
  2nd harmonic about -15 dB); the rest is a masenqo.
- Different register: insert plays B4-E5, the rest stays in C4-A4.
- Low-frequency tones at 30-95 Hz under every inserted note, 20-30 dB above
  the rest of the file in the 20-150 Hz band.
- The original has a note-dependent DC drift below 10 Hz; the insert has none.

There is no click or level jump at the seams, as the task says.

## Method

1. About 40 features every 50 ms: f0, tuning, harmonic ratios, MFCCs,
   sub-fundamental energy, infrasonic/DC level, noise floor, 50/60 Hz hum,
   skewness, crest factor, envelope modulation, pitch jitter.
2. Median-filter, rank-gaussianise and whiten the features.
3. Exhaustive search for the single contiguous block (1.5 s to 40 s, at most
   half the file) whose frames differ most from all other frames.

The detector does not assume which direction any feature moves, so it also
works when the roles of the two instruments are swapped.

## Validation

| Set | Passed |
|---|---|
| Re-spliced train material (`validation/harness.py`) | 49 / 49 |
| Harder re-splices: short/long, edges, noise, fades (`harness2.py`) | 96 / 100 |
| Synthetic masenqo/washint, different instrument (`synth*.py`) | 46 / 46 |
| Synthetic, same instrument | 87 / 92 |
| Train with noise, filters, resampling, other sample formats (`perturb.py`) | all |
| Feature-group ablation, prime length, reversed audio (`control.py`) | all |

Known weak spot: an insert played on the same instrument through the same
recording chain. Without the low-frequency cues the start estimate on train
lands 1.0 s late (still inside tolerance).

## Files

- `main.py` - the submission.
- `train-output.csv` - ground truth for the train recording.
- `analysis/` - exploration scripts (spectra, pitch track, plots).
- `validation/` - test-case generators and evaluation runners.
- `figures/` - spectrogram overview, splice zoom, DC drift vs pitch.

`train-input.wav` is the organisers' file and is not included. The scripts
in `analysis/` and `validation/` use the absolute paths of the environment
they were written in (`/home/claude/...`); adjust them before re-running.
