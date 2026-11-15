"""Stress tests for a fixed submitted policy.

A. Small test sets, using angkol's REAL catches: random subsets of the 400 practice days.
B. Fish lost at sea (hold > 0 when the day ends) in the nominal lagoon.
C. Sensitivity: the same fixed policy in lagoons whose reef bite / tide / seagrass bite
   differ from lagoon.py (the logbook cannot confirm reef bite or tide: angkol never went
   past the passage). Angkol is approximated by a clone (majority action + 5.7% noise).

Usage: python -I stress.py <task_dir> <policy.csv>
"""
import csv
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

TASK, POL = sys.argv[1], sys.argv[2]
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402

pol = L.read_policy(POL)
ref = json.load(open(f"{TASK}/train-output.json"))
real = dict(zip(ref["days"], ref["angkol_catch"]))

# ---------- A ----------
ours_practice = {d: L.play_day(pol, d)[0] for d in ref["days"]}
rng = random.Random(1)
print("A. random subsets of the 400 practice days (angkol's real catches, same days):")
for n in (1, 3, 5, 10, 20, 50, 100):
    fails, trials = 0, 20000
    for _ in range(trials):
        days = rng.sample(ref["days"], n)
        a = sum(real[d] for d in days)
        o = sum(ours_practice[d] for d in days)
        fails += o < 1.5 * a
    print(f"   {n:>3} days: P(ratio < 1.5) = {fails / trials:.4%}")

# ---------- B ----------
end_hold = Counter()
for d in range(401, 10401):
    day = L.Lagoon(d)
    s, done = day.reset(), False
    while not done:
        s, _, done = day.step(pol[s])
    end_hold[day.hold] += 1  # fish still aboard when the day ends = fish lost
print("\nB. fish lost at sea (hold when the day ends) over 10,000 unseen days:", dict(end_hold))

# ---------- C ----------
rows = list(csv.DictReader(open(f"{TASK}/dataset/logbook.csv")))
acts = defaultdict(Counter)
for rec in rows:
    acts[(int(rec["row"]), int(rec["col"]), int(rec["hold"]), int(rec["steps_left"]))][rec["action"]] += 1
major = {s: c.most_common(1)[0][0] for s, c in acts.items() if sum(c.values()) >= 3}
home = L.START


def angkol_greedy(s):
    if s in major:
        return major[s]
    r, c, h, t = s
    dist = abs(r - home[0]) + abs(c - home[1])
    if L.GRID[r][c] == "s" and h < 6 and t > dist + 2:
        return "FISH"
    if h > 0 or t <= dist + 2:
        if r < home[0] and L.is_water(r + 1, c):
            return "SOUTH"
        return "WEST" if c > home[1] else "EAST" if c < home[1] else "WAIT"
    if r > 3:
        return "NORTH"
    return "EAST" if c < 4 else "WEST" if c > 4 else "NORTH"


def angkol_day(seed):
    noise = random.Random(7_000_000 + seed)
    return L.play_day(lambda s: noise.choice(L.ACTIONS) if noise.random() < 0.057 else angkol_greedy(s), seed)[0]


nominal = (dict(L.BITE_CHANCE), L.TIDE_CHANCE)
print("\nC. same fixed policy, different hidden constants (3,000 unseen days each):")
print("   reef bite  tide  seagrass bite |  ours   angkol~  ratio")
for t_bite, tide, s_bite in [(0.9, 0.4, 0.2), (0.75, 0.4, 0.2), (0.6, 0.4, 0.2), (0.5, 0.4, 0.2),
                             (0.9, 0.6, 0.2), (0.9, 0.8, 0.2), (0.6, 0.6, 0.2), (0.9, 0.4, 0.3)]:
    L.BITE_CHANCE.clear()
    L.BITE_CHANCE.update({"s": s_bite, "T": t_bite})
    L.TIDE_CHANCE = tide
    seeds = range(50_001, 53_001)
    o = st.mean(L.play_day(pol, d)[0] for d in seeds)
    a = st.mean(angkol_day(d) for d in seeds)
    flag = "" if o >= 1.5 * a else "   <-- below 1.5x"
    print(f"   {t_bite:>9}  {tide:>4}  {s_bite:>13} | {o:5.2f}   {a:5.2f}   {o / a:5.2f}x{flag}")
L.BITE_CHANCE.clear()
L.BITE_CHANCE.update(nominal[0])
L.TIDE_CHANCE = nominal[1]
