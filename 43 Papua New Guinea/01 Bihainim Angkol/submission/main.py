"""main.py for task_papua_new_guinea (Bihainim Angkol / Follow Your Uncle).

Reads test-input.csv (row,col,hold,steps_left) from the current directory and
writes test-output.csv (row,col,hold,steps_left,action) next to it, one row per
input row, in the same order.

The agent
---------
Angkol's logbook replays exactly under the lagoon's rules with seed = day, so
the lagoon is a fully known finite-horizon MDP:
    state  (row, col, hold, steps_left): 36 water cells x 7 holds x 24 steps = 6,048
    chance bite on s (0.20) or T (0.90) when fishing, tide on ~ (0.40, one cell south)
    reward fish landed on the beach; fish still aboard at the end are lost
Angkol only ever fishes the seagrass (20% bites). The outer reef (90% bites)
lies past the reef passage, where the tide pushes canoes back, and he never
goes there. Backward induction over all 6,048 states gives the policy that
maximises the expected catch. That policy crosses the passage, fills the hold
on the drop-off and comes home: about 5.96 fish a day, against angkol's 2.49.

Ties in expected catch are broken by landing the fish as early as possible
(lexicographic DP), so a full canoe heads straight home instead of idling at
sea. Any tie that is left goes to WAIT, then to a fixed action order, so once
nothing more can be caught the canoe stays on the beach.

Standard library only; the whole solve takes well under a second.
"""

import csv

# ---- the lagoon (identical to lagoon.py) ---------------------------------
MAP = (
    ". . . . T T .",
    "# # # # ~ # #",
    ". . . . s s .",
    ". . . . s s .",
    ". . . . . . .",
    ". . H . . . .",
)
GRID = tuple(tuple(line.split()) for line in MAP)
ROWS, COLS = len(GRID), len(GRID[0])
BEACH, PASSAGE, REEF = "H", "~", "#"
STEPS_PER_DAY = 24
HOLD_CAPACITY = 6
BITE_CHANCE = {"s": 0.20, "T": 0.90}
TIDE_CHANCE = 0.40
TIDE_PUSH = (1, 0)

ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST", "FISH", "WAIT")
MOVES = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}

TIE = 1e-9  # expected catches closer than this count as equal
PREFERENCE = ("WAIT", "NORTH", "SOUTH", "EAST", "WEST", "FISH")  # last tie-break: idle if nothing to gain


def is_water(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS and GRID[row][col] != REEF


def advance(row, col, hold, action, bite, tide):
    """One step with its chance events decided. Returns (row, col, hold, landed)."""
    if action in MOVES:
        dr, dc = MOVES[action]
        if is_water(row + dr, col + dc):
            row, col = row + dr, col + dc
    elif action == "FISH":
        if GRID[row][col] in BITE_CHANCE and hold < HOLD_CAPACITY and bite:
            hold += 1
    if GRID[row][col] == PASSAGE and tide:
        dr, dc = TIDE_PUSH
        if is_water(row + dr, col + dc):
            row, col = row + dr, col + dc
    landed = 0
    if GRID[row][col] == BEACH:
        landed, hold = hold, 0
    return row, col, hold, landed


def outcomes(row, col, hold, action):
    """[(probability, next_row, next_col, next_hold, landed)] for one decision."""
    p_bite = BITE_CHANCE.get(GRID[row][col], 0.0)
    result = []
    for bite, pb in ((True, p_bite), (False, 1.0 - p_bite)):
        if pb == 0.0:
            continue
        for tide, pt in ((True, TIDE_CHANCE), (False, 1.0 - TIDE_CHANCE)):
            result.append((pb * pt, *advance(row, col, hold, action, bite, tide)))
    return result


def solve():
    """Backward induction. Returns ({state: action}, expected catch from the beach)."""
    cells = [(r, c) for r in range(ROWS) for c in range(COLS) if is_water(r, c)]
    trans = {
        (r, c, h, a): outcomes(r, c, h, a)
        for (r, c) in cells
        for h in range(HOLD_CAPACITY + 1)
        for a in ACTIONS
    }
    # catch[s]: expected fish landed from s (primary objective)
    # early[s]: expected sum of fish x steps_left when landed (secondary: land sooner)
    catch = {(r, c, h): 0.0 for (r, c) in cells for h in range(HOLD_CAPACITY + 1)}
    early = dict(catch)
    policy = {}
    for t in range(1, STEPS_PER_DAY + 1):
        new_catch, new_early = {}, {}
        for (r, c) in cells:
            for h in range(HOLD_CAPACITY + 1):
                q = {}
                for a in ACTIONS:
                    q1 = q2 = 0.0
                    for p, r2, c2, h2, landed in trans[(r, c, h, a)]:
                        q1 += p * (landed + catch[(r2, c2, h2)])
                        q2 += p * (landed * t + early[(r2, c2, h2)])
                    q[a] = (q1, q2)
                best1 = max(v[0] for v in q.values())
                tied = [a for a in PREFERENCE if q[a][0] >= best1 - TIE]
                best = max(tied, key=lambda a: q[a][1])  # first in PREFERENCE order on a tie
                policy[(r, c, h, t)] = best
                new_catch[(r, c, h)], new_early[(r, c, h)] = q[best]
        catch, early = new_catch, new_early
    start = next((r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == BEACH)
    return policy, catch[(*start, 0)]


FIELDS = ("row", "col", "hold", "steps_left")


def parse_state(rec):
    """(row, col, hold, steps_left) as ints, or the raw values if they do not parse."""
    raw = tuple((rec.get(k) or "").strip() for k in FIELDS)
    try:
        return tuple(int(float(x)) for x in raw)
    except (ValueError, OverflowError):
        return raw


def main():
    policy, _ = solve()
    with open("test-input.csv", newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames:
            reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
        records = list(reader)
    with open("test-output.csv", "w", newline="") as dst:
        out = csv.writer(dst)
        out.writerow([*FIELDS, "action"])
        for rec in records:
            state = parse_state(rec)
            out.writerow([*state, policy.get(state, "WAIT")])


if __name__ == "__main__":
    main()
