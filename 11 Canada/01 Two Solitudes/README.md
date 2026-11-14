**SIGMOID_SNOWFLAKE**

The synthetic corpus ties hazards to language (flood/winter/wildfire mostly French), to header places, to boilerplate sentences and to planted sentences about other hazards; the first model learns those, not the hazards (the provided first model opens a decoy door, SNOWBLIND).
`train.py` finds the hazard sentences, rebuilds balanced reports from them with random headers and boilerplate, swaps hazard words between English and French, and averages eight logistic regressions (b = 0, small month term).
The sealed fingerprints are used only as unlabeled counts, to zero cells that are hash collisions with common real words and to put the eight models on the same scale before averaging; no labels, no lookups.
`python train.py` writes `model.npz`; `python ../task_canada/open_vault.py model.npz` releases the flag. The notebook has the corpus analysis, held-out checks and the vault log (12 runs).
