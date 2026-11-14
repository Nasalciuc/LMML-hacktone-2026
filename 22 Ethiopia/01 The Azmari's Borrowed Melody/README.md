# The Azmari's Borrowed Melody

Train: 14.95-20.95, true 14.96-20.965. Platform: first version (ec2b674) failed test-3, second (b061ff3) failed test-2. This two-pass version hasn't been submitted yet.

`python main.py test-1-input.wav` writes `detection.csv`. Needs numpy + scipy only, runs in ~6 s.
First pass is ec2b674 unchanged (37 chain/colour tracks, 1.5-20 s stretches), and it gives identical answers. Its answer stands when the second pass agrees within 1.5 s, or when it beats every non-overlapping stretch by 1.5x or more. Train scores 2.6x; train-like synthetic splices score 1.56x and up.
Second pass is b061ff3 without the short-stretch discount. That discount was the only change that pushed edges out on train-like splices (worst edge 1.17 s vs 0.39 s without it), so it's the likely test-2 culprit. The second pass adds Nyquist-band and floor-shape tracks, tuning, 1-30 s stretches, the end-of-file swap and 25 ms edge refinement.
`python validate.py ../task_ethiopia 20 3 -v` has two sets, both built from train's host with the real splice cut out. "realistic" uses the real phrase, or host pieces with no sub-20 Hz drift and a 20-150 Hz floor raised by 3-25 dB; 3-9 s, 40 % at an edge. "tough" uses resample, codec, EQ, pitch, reverb, hum and hiss changes; 1-30 s.
Realistic: first 80/80, second 80/80, answer 80/80. Tough: first 111/180, second 149/180, answer 148/180.
