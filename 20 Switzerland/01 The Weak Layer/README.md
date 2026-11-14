# The Weak Layer

Submit `main.py`. Reads `test-input.csv`, writes `test-output.txt`.

Default `DecisionTreeClassifier(random_state=0)`, then read the split features from the root down (the tree is a chain: each question has a leaf on one side and the next question on the other). First letters of those features = the word. If the tree ever branches, it follows the path to the deepest "released" leaf.

Practice table: depth_hoar_pct → elevation_m → mean_grain_roundness → overburden_load_kpa → **DEMO**, same for random_state 0/7/123 and with the columns reversed.
