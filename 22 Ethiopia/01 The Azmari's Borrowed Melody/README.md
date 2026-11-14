# The Azmari's Borrowed Melody

Train: detected 14.95-20.95, true 14.96-20.965 (errors 0.01 s / 0.01 s, tolerance 1.5 s). Synthetic splices: 148/150 within 1.5 s. Flag: not yet, comes from the checker.

`python main.py test-1-input.wav` writes `detection.csv`. Needs numpy + scipy only, runs in ~5 s.
A pasted-in phrase brings its own recording chain. The script tracks about 40 things on 0.4 s frames: DC and sub-8 Hz drift, rumble, hiss floor, high-band tilt and flatness, 50/60 Hz hum, envelope dips and spectral flux (reverb), and mel-band colour. It decorrelates them and picks the 1.5-20 s stretch whose mean differs most from the rest.
`python validate.py ../task_ethiopia 25` cuts the real splice out of the training file and pastes 2.5-8 s pieces back in at random spots (25 per kind). Kinds: the real foreign phrase 25/25; host pieces high-passed with new hiss 25/25, with room reverb 23/25, with mains hum 25/25, low-passed 25/25, with extra hiss 25/25. Median edge error is 0.04-0.17 s.
