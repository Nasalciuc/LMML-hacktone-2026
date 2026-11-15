"""Replay angkol's logbook against lagoon.py and describe his behaviour.

Usage: python -I analyze_logbook.py <task_dir>
"""
import csv
import json
import sys
from collections import Counter, defaultdict

TASK = sys.argv[1]
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402

rows = list(csv.DictReader(open(f"{TASK}/dataset/logbook.csv")))
ref = json.load(open(f"{TASK}/train-output.json"))

by_day = defaultdict(list)
for r in rows:
    by_day[int(r["day"])].append(r)

# 1) Is the logbook exactly this simulator with seed = day?
mismatch = 0
catches = {}
for day, recs in by_day.items():
    lag = L.Lagoon(day)
    s = lag.reset()
    total = 0
    for rec in recs:
        st = (int(rec["row"]), int(rec["col"]), int(rec["hold"]), int(rec["steps_left"]))
        if st != s:
            mismatch += 1
            break
        s, landed, done = lag.step(rec["action"])
        if landed != int(rec["reward"]):
            mismatch += 1
            break
        total += landed
    catches[day] = total
print("days in logbook:", len(by_day), "| steps per day:", Counter(len(v) for v in by_day.values()))
print("days whose replay diverges from lagoon.py(seed=day):", mismatch)
ref_c = dict(zip(ref["days"], ref["angkol_catch"]))
print("catch matches train-output.json on all days:", all(catches[d] == ref_c[d] for d in ref_c))
print("angkol mean:", sum(catches.values()) / len(catches), "| reference:", ref["angkol_mean"])
print("catch distribution:", sorted(Counter(catches.values()).items()))

# 2) Is his policy deterministic in the state?
acts = defaultdict(Counter)
for r in rows:
    st = (int(r["row"]), int(r["col"]), int(r["hold"]), int(r["steps_left"]))
    acts[st][r["action"]] += 1
conflicts = {s: c for s, c in acts.items() if len(c) > 1}
print("distinct states visited:", len(acts), "| states with >1 action:", len(conflicts))
for s, c in list(conflicts.items())[:10]:
    print("   ", s, dict(c))

# 3) Where does he go and what does he do?
cells = Counter((int(r["row"]), int(r["col"])) for r in rows)
print("cells visited (row,col): count")
for (rr, cc), n in sorted(cells.items()):
    print(f"   ({rr},{cc}) {L.GRID[rr][cc]}  {n}")
print("actions:", Counter(r["action"] for r in rows))
fish_by_ground = Counter(L.GRID[int(r["row"])][int(r["col"])] for r in rows if r["action"] == "FISH")
print("FISH actions by ground:", fish_by_ground)

# 4) When does he turn for home? (first step he heads back with fish aboard)
turn = []
for day, recs in by_day.items():
    for rec in recs:
        g = L.GRID[int(rec["row"])][int(rec["col"])]
        if g == "s" and rec["action"] in ("SOUTH", "WEST"):
            turn.append((int(rec["hold"]), int(rec["steps_left"])))
            break
print("(hold, steps_left) when he leaves the seagrass, most common:", Counter(turn).most_common(15))
home = []
for day, recs in by_day.items():
    land = [int(r["steps_left"]) for r in recs if int(r["reward"]) > 0]
    home.append(land[0] if land else None)
print("steps_left at the step he lands (most common):", Counter(home).most_common(10))
