# Bihainim Angkol

Submit `main.py` (made by `python solve.py`, which also writes `policy.csv`).

| | fish per day |
|---|---|
| angkol, logbook days 1-400 | 2.490 |
| this policy, days 1-400 | 5.963 (2.39x) |
| this policy, days 401-5400 | 5.965 |

Angkol only fishes the seagrass flats (4,954 of his FISH actions are on `s`, none on `T`), where a bite is 20 %. The outer reef `T` bites 90 % but sits past the reef passage, where the tide pushes the canoe back south 40 % of the time.

All rules and odds are in `lagoon.py`, so the best action for every state can be computed exactly: backward induction over (row, col, hold, steps_left), 24 steps, expected fish landed as the value. The result: go out through the passage, fill the hold at `T` (about 7 casts for 6 fish), come back with the tide and land on the beach. Expected catch from the beach: 5.957 fish per day.

Check:

```
cp ../task_papua_new_guinea/train-input.csv test-input.csv
python main.py
python ../task_papua_new_guinea/lagoon.py --policy test-output.csv --compare ../task_papua_new_guinea/train-output.json
```

`main.py` uses only the standard library and runs in about 0.1 s.
