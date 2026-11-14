# South Africa — Ngiyakuzwa?

Flag: not found yet. The word after *ikhodi* has to be found by ear in the cleaned audio.

## What to play, in order

1. `ikhodi_candidates/rank01_*.wav` … `rank10_*.wav` — 10 s clips (3 s before a likely
   "kh" burst, 7 s after). Listen for *i-KHO-di* and write down the next word.
   `candidates.csv` has the times. The ranking is a rough acoustic guess, not a detection.
2. `cleaned/voice_note_clean.wav` — whole note, horns and crowd floor removed. Best overall.
3. `cleaned/voice_note_clean_strong.wav` — harder noise removal. Less crowd, more "underwater"
   sound on the voice. Try it where (2) is still too noisy.
4. `cleaned/voice_note_clean_voiceband.wav` — (2) limited to 250–3800 Hz.
5. `segments/NNN_<start>-<end>.wav` — 138 phrases cut from (2), in order, with a
   spectrogram `.png` each (cleaned on top, original below). `segments/index.csv` lists them.
   If no candidate has *ikhodi*, go through these in order.

The candidate clips and segments are taken from `voice_note_clean.wav`; times in file names are
positions in the original note.

## Noise found in the note

- Two horns (vuvuzela-like), steady harmonic combs at 217.5 Hz × k and 233.1 Hz × k, switching
  on and off in blocks of a few seconds (69 switches). They mask most of the voice band.
- Other steady whistles / tones, some gliding slowly.
- Crowd roar (broadband) and a hiss band around 5–7 kHz; short claps.

## Cleaning (`clean.py`)

STFT, 64 ms window, 8 ms hop.

- Noise floor per frequency bin = running 25th percentile over 3 s. Speech syllables are short
  and move in frequency, so they stay above it; horns and roar do not.
- Steady tonal lines removed: a bin that is a spectral peak for most of 0.6 s is noise (speech
  harmonics glide). Plus the two horn combs explicitly, in frames where the horn is on.
  Those bins are cut by ~30 dB.
- Wiener-type gain (over-subtraction 2, floor −24 dB), smoothed over 3 bins × 5 frames.
- 120–7000 Hz kept.

Strong version: over-subtraction 4, floor −34 dB, 35th percentile, 200–5000 Hz.

## Segments and candidates (`segment.py`)

- Speech activity: cleaned energy in 250–3500 Hz more than 6 dB above its 6 s running 20th
  percentile. Gaps < 0.25 s are joined into words, < 0.7 s into phrases, phrases ≥ 0.4 s kept,
  0.25 s padding. Result: 267 s active, 332 word chunks, 138 phrases.
- *ikhodi* score per word chunk: front vowel (i, high F2) → short dip → 40–120 ms aspiration
  (2.5–6.5 kHz dominant, the "kh") → back vowel (o, low F2) → closure → front vowel (i).
  Bursts within 0.2 s of a horn on/off switch are ignored (the switch clicks look like bursts).
  The score is weak: top 10 are close together (10.4–13.1), so treat the order loosely.

## Run

```
python -m venv .venv
.venv\Scripts\pip install numpy scipy matplotlib soundfile
.venv\Scripts\python clean.py      # ~3 min, writes cleaned/
.venv\Scripts\python segment.py    # ~3 min, writes segments/ and ikhodi_candidates/
```

Input: `../task_south_african_republic/voice_note.wav`. No speech recognition model is used.
