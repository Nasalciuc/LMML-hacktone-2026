"""Exact-arithmetic (fractions.Fraction) check of a submitted policy.

1. V*   : exact optimal expected catch, by backward induction with exact probabilities.
2. V^pi : exact expected catch of the submitted policy (policy evaluation).
3. V_sg : exact optimum for a canoe that never crosses the reef passage (any angkol-style
          fisher, however good) -> the largest catch angkol could ever average.
Uses lagoon.py's own _advance for every transition.

Usage: python -I exact_check.py <task_dir> <policy.csv>
"""
import sys
from fractions import Fraction as F

TASK, POL = sys.argv[1], sys.argv[2]
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402

BITE = {"s": F(1, 5), "T": F(9, 10)}
TIDE = F(2, 5)
assert {k: float(v) for k, v in BITE.items()} == L.BITE_CHANCE and float(TIDE) == L.TIDE_CHANCE
H, T = L.HOLD_CAPACITY, L.STEPS_PER_DAY
water = [(r, c) for r in range(L.ROWS) for c in range(L.COLS) if L.is_water(r, c)]
PASSAGE = next((r, c) for (r, c) in water if L.GRID[r][c] == L.PASSAGE)


def outcomes(r, c, h, a):
    pb = BITE.get(L.GRID[r][c], F(0))
    res = []
    for bite, p1 in ((True, pb), (False, 1 - pb)):
        for tide, p2 in ((True, TIDE), (False, 1 - TIDE)):
            if p1 * p2:
                res.append((p1 * p2, *L._advance(r, c, h, a, bite, tide)))
    return res


TRANS = {(r, c, h, a): outcomes(r, c, h, a) for (r, c) in water for h in range(H + 1) for a in L.ACTIONS}
for v in TRANS.values():
    assert sum(p for p, *_ in v) == 1


def q_value(V, r, c, h, a):
    return sum(p * (land + V[(r2, c2, h2)]) for p, r2, c2, h2, land in TRANS[(r, c, h, a)])


def optimum(allowed=lambda r, c, h, a: True):
    V = {(r, c, h): F(0) for (r, c) in water for h in range(H + 1)}
    allV, argmax = {}, {}
    for t in range(1, T + 1):
        nV = {}
        for (r, c) in water:
            for h in range(H + 1):
                qs = {a: q_value(V, r, c, h, a) for a in L.ACTIONS if allowed(r, c, h, a)}
                best = max(qs.values())
                nV[(r, c, h)] = best
                argmax[(r, c, h, t)] = {a for a, q in qs.items() if q == best}
        V = nV
        allV[t] = V
    return allV, argmax


def evaluate(policy):
    V = {(r, c, h): F(0) for (r, c) in water for h in range(H + 1)}
    allV = {}
    for t in range(1, T + 1):
        V = {(r, c, h): q_value(V, r, c, h, policy[(r, c, h, t)]) for (r, c) in water for h in range(H + 1)}
        allV[t] = V
    return allV


start = (*L.START, 0)
Vstar, argmax = optimum()
pol = L.read_policy(POL)
Vpi = evaluate(pol)
print(f"V*(start)   = {float(Vstar[T][start]):.15f}   exact = {Vstar[T][start]}")
print(f"V^pi(start) = {float(Vpi[T][start]):.15f}   identical to V*: {Vpi[T][start] == Vstar[T][start]}")
gaps = [(Vstar[t][s] - Vpi[t][s], (*s, t)) for t in range(1, T + 1) for s in Vstar[t]]
worse = [g for g in gaps if g[0] != 0]
print(f"states (all 6,048) where the policy is below the exact optimum: {len(worse)}"
      + (f", largest gap {float(max(worse)[0]):.3e} at {max(worse)[1]}" if worse else ""))
not_argmax = [s for s, acts in argmax.items() if pol[s] not in acts]
print(f"states whose action is not an exact argmax: {len(not_argmax)}")


def never_cross(r, c, h, a):
    if a in L.MOVES:
        dr, dc = L.MOVES[a]
        return (r + dr, c + dc) != PASSAGE
    return True


Vsg, _ = optimum(never_cross)
best_sg = Vsg[T][start]
print(f"\nbest possible fisher that never crosses the passage: {float(best_sg):.4f} fish/day")
print(f"our policy / that bound = {float(Vstar[T][start] / best_sg):.3f}x  (needs >= 1.5)")
print(f"angkol's actual logbook mean 2.490 -> {float(Vstar[T][start]) / 2.49:.3f}x")
