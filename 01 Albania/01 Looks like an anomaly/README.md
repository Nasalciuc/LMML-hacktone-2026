# Looks like an anomaly!

Submit `detector.py`. Only the standard library, so it runs on the organizers' pinned setup.

Local score on the training stream (`python evaluate.py`): **97.6 / 100** (pass is 70).

| | |
|---|---|
| F1 | 0.975 (5 false alarms, 9 misses out of 287) |
| point recall | 1.00 |
| drift recall | 0.96 |
| collective recall | 0.98 |

How it works, per reading:

- z-score of each sensor against a slowly updated normal level (starts from the training stats, re-centred on the first 150 readings of the stream);
- **point**: any sensor above 6 sd;
- **drift**: short moving average of a sensor above 3 sd, and the current reading still above 2 sd;
- **collective**: average squared z over all sensors high, smoothed and current;
- the normal level only learns from readings that look normal.

Also checked on the second half alone, a start mid-event, levels shifted by 2 and 4 sd, extra noise and the reversed stream: lowest score 89.6.
