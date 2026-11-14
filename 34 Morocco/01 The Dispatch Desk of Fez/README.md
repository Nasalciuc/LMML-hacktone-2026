# The Dispatch Desk of Fez

Train (400 requests): quality 0.8275 (gate 0.80), cost 7000 (gate 8000). That's 17.5 dirham a request against a budget of 20. Flag: not yet, comes from the checker.

`main.py` uses only the standard library. It reads `test-input.csv` and writes `routing_log.csv`.
The rules come from the recorded answers, family by family. Plain board prices go to the apprentice. So do small multiplications, and shortest-walk questions where a detour beats the direct path (both guides get those wrong). Takings after a per-crate fee also go to the apprentice (the master gets only about half of them right, so they're not worth the money). Everything else goes to the master.
The best possible score on train is 0.88, since both guides miss 48 of the 400.
`python score.py ../task_morroco` prints per-family numbers and a stress test that reweights the family mix.
