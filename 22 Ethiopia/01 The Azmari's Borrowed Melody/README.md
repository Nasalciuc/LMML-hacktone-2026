# The Azmari's Borrowed Melody

Train: detected 14.84-21.01, true 14.96-20.965 (errors 0.12 s / 0.05 s, tolerance 1.5 s). Synthetic splices: 166/170 within 1.5 s, plus 10 that can't be told apart (see below). Platform, first version: test-1 and test-2 passed, test-3 failed. This version hasn't been resubmitted yet.

`python main.py test-1-input.wav` writes `detection.csv`. Needs numpy + scipy only, runs in ~5 s.
It tracks about 70 things on 0.4 s frames: DC and drift, rumble, hum, hiss floor and its shape, the top of the spectrum up to Nyquist (resampling and codec cutoffs), reverb (dips, flux), tuning against the pitch grid, and mel colour. It decorrelates them and picks the 1-30 s stretch that differs most from the rest (short stretches are discounted). If that stretch sits at one end of the file and the remainder is shorter but nearly as different, it reports the remainder. Each edge is then refined on a 25 ms grid.
`python validate.py ../task_ethiopia 20 2 -v` cuts the real splice out of the training file. It then pastes pieces back in, RMS-matched, 1-30 s long, a third of them at an edge, into rotated or chunk-shuffled hosts. Kinds: the real foreign phrase, a resample round trip, a codec-style low-pass, mic EQ, a 0.5-3 % pitch change, high-pass + new hiss, reverb, hum, extra hiss.
Results (20 per kind): foreign 20, resample 17, codec 18, eq 18, pitch 17, clean 19, reverb 20, hum 19, noisier 18. By length: 1-3 s 12/13, 3-10 s 50/50, 10-30 s 104/117.
10 of the 14 misses are splices longer than half the file that touch an edge. There the remainder differs from the rest by exactly as much, and the script deliberately returns the shorter side. The real misses are 2 EQ-only cases and 2 near-edge 25 s splices.
