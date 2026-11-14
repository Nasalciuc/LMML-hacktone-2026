# exact best policy for the lagoon: backward induction over (row, col, hold, steps_left)
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.join(HERE, "..", "task_papua_new_guinea")
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402

V = {}       # expected fish landed from this state to the end of the day
POLICY = {}


def outcomes(r, c, h, a):
    """(prob, next_row, next_col, next_hold, landed) for one decision."""
    p_bite = L.BITE_CHANCE.get(L.GRID[r][c], 0.0) if a == "FISH" else 0.0
    out = []
    for bite, pb in ((True, p_bite), (False, 1 - p_bite)):
        if pb == 0:
            continue
        # tide only matters if we end the step on the passage
        r1, c1, h1, _ = L._advance(r, c, h, a, bite, False)
        if L.GRID[r1][c1] == L.PASSAGE:
            for tide, pt in ((True, L.TIDE_CHANCE), (False, 1 - L.TIDE_CHANCE)):
                out.append((pb * pt, *L._advance(r, c, h, a, bite, tide)))
        else:
            out.append((pb, *L._advance(r, c, h, a, bite, False)))
    return out


for t in range(1, L.STEPS_PER_DAY + 1):
    for (r, c, h, tt) in L.all_states():
        if tt != t:
            continue
        best, best_a = -1.0, None
        for a in ("FISH", "NORTH", "SOUTH", "EAST", "WEST", "WAIT"):
            q = sum(p * (land + (V[(r2, c2, h2, t - 1)] if t > 1 else 0.0))
                    for p, r2, c2, h2, land in outcomes(r, c, h, a))
            if q > best + 1e-12:
                best, best_a = q, a
        V[(r, c, h, t)] = best
        POLICY[(r, c, h, t)] = best_a

start = (*L.START, 0, L.STEPS_PER_DAY)
print(f"expected fish per day from the beach: {V[start]:.3f}")

# angkol, from the logbook
rows = list(csv.DictReader(open(os.path.join(TASK, "dataset", "logbook.csv"))))
days = Counter()
for x in rows:
    days[x["day"]] += int(x["reward"])
print(f"angkol: {sum(days.values()) / len(days):.3f} fish per day over {len(days)} days")
where = Counter(L.GRID[int(x['row'])][int(x['col'])] for x in rows if x["action"] == "FISH")
print("angkol fishes on:", dict(where))

days = list(range(1, 401))
mine = sum(L.play_day(POLICY, d)[0] for d in days) / len(days)
print(f"policy on the 400 logbook days: {mine:.3f} fish per day")
_, traj = L.play_day(POLICY, 1)
print("day 1:", " ".join(a for _, a, _ in traj))

L.write_policy(POLICY, os.path.join(HERE, "policy.csv"))
L.export_main(POLICY, os.path.join(HERE, "main.py"))
