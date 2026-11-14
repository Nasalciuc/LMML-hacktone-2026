# Just a siblings' thing

Flag: `SIGMOID_ARES`

`python evaluate_oracles.py` (torch 2.6, weights loaded with `weights_only=True`). Output in [run_output.txt](run_output.txt), figure [oracles.png](oracles.png).

| | accuracy | recall | precision | F1 | missed threats | false alarms |
|---|---|---|---|---|---|---|
| ARES | 0.95 | **0.95** | 0.68 | 0.79 | **5** | 45 |
| ATHENA | 0.95 | 0.75 | 0.75 | 0.75 | 25 | 25 |
| APOLLO | 0.95 | 0.55 | 0.92 | 0.69 | 45 | 5 |

Same accuracy, but a missed catastrophic threat costs far more than a false alarm, so recall decides: Ares misses 5 of 100 threats, Apollo misses 45.
