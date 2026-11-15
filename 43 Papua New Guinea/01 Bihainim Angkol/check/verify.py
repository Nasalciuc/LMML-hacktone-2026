"""Verify a main.py output against lagoon.py, and estimate the ratio on unseen days.

Usage: python -I verify.py <task_dir> <test-output.csv>
"""
import csv
import random
import statistics as st
import sys
from collections import Counter, defaultdict

TASK, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, TASK)
import lagoon as L  # noqa: E402

# ---------- 1. independent lexicographic DP using lagoon's own _advance ----------
water = [(r, c) for r in range(L.ROWS) for c in range(L.COLS) if L.is_water(r, c)]
H, T = L.HOLD_CAPACITY, L.STEPS_PER_DAY


def dp(lexicographic):
    v1 = {(r, c, h): 0.0 for (r, c) in water for h in range(H + 1)}
    v2 = dict(v1)
    pol = {}
    for t in range(1, T + 1):
        n1, n2 = {}, {}
        for (r, c) in water:
            for h in range(H + 1):
                pb = L.BITE_CHANCE.get(L.GRID[r][c], 0.0)
                q = {}
                for a in L.ACTIONS:
                    q1 = q2 = 0.0
                    for bite, p1 in ((True, pb), (False, 1 - pb)):
                        for tide, p2 in ((True, L.TIDE_CHANCE), (False, 1 - L.TIDE_CHANCE)):
                            if p1 * p2 == 0:
                                continue
                            r2, c2, h2, land = L._advance(r, c, h, a, bite, tide)
                            q1 += p1 * p2 * (land + v1[(r2, c2, h2)])
                            q2 += p1 * p2 * (land * t + v2[(r2, c2, h2)])
                    q[a] = (q1, q2)
                m = max(x[0] for x in q.values())
                order = ("WAIT", "NORTH", "SOUTH", "EAST", "WEST", "FISH") if lexicographic else L.ACTIONS
                cand = [a for a in order if q[a][0] >= m - 1e-9]
                a = max(cand, key=lambda x: q[x][1]) if lexicographic else cand[0]
                pol[(r, c, h, t)] = a
                n1[(r, c, h)], n2[(r, c, h)] = q[a]
        v1, v2 = n1, n2
    return pol, v1[(*L.START, 0)]


ref_pol, ref_v = dp(True)
plain_pol, plain_v = dp(False)
sub = L.read_policy(OUT)
diff = [s for s in L.all_states() if sub.get(s) != ref_pol[s]]
print(f"expected catch: lexicographic {ref_v:.6f} | plain argmax {plain_v:.6f}")
print(f"main.py output vs independent DP: {len(diff)} differing states out of {len(L.all_states())}")
print("validator problems:", L.check_policy_file(OUT) or "none")

# ---------- 2. angkol clone (majority action + estimated noise) ----------
rows = list(csv.DictReader(open(f"{TASK}/dataset/logbook.csv")))
acts = defaultdict(Counter)
for r in rows:
    acts[(int(r["row"]), int(r["col"]), int(r["hold"]), int(r["steps_left"]))][r["action"]] += 1
major = {s: c.most_common(1)[0][0] for s, c in acts.items()}
busy = [c for c in acts.values() if sum(c.values()) >= 100]
dev = sum(sum(c.values()) - c.most_common(1)[0][1] for c in busy) / sum(sum(c.values()) for c in busy)
eps = dev * 6 / 5  # uniform random action w.p. eps -> deviates w.p. eps*5/6
print(f"angkol noise: {dev:.3%} of decisions off his usual action -> eps ~ {eps:.3f}")


def home_dist(r, c):
    return abs(r - L.START[0]) + abs(c - L.START[1])


def angkol_greedy(s):
    if s in major and sum(acts[s].values()) >= 3:
        return major[s]
    r, c, h, t = s
    if L.GRID[r][c] == "s" and h < H and t > home_dist(r, c) + 2:
        return "FISH"
    if h > 0 or t <= home_dist(r, c) + 2:  # head home: south first, then west/east
        if r < L.START[0] and L.is_water(r + 1, c):
            return "SOUTH"
        return "WEST" if c > L.START[1] else "EAST" if c < L.START[1] else "WAIT"
    if r > 3:
        return "NORTH"
    return "EAST" if c < 4 else "WEST" if c > 4 else "NORTH"


def angkol_day(seed, noise_seed):
    rng = random.Random(noise_seed)
    return L.play_day(lambda s: rng.choice(L.ACTIONS) if rng.random() < eps else angkol_greedy(s), seed)[0]


practice = [angkol_day(d, 10_000_000 + d) for d in range(1, 401)]
print(f"angkol clone on the 400 practice days: {st.mean(practice):.3f} (real logbook: 2.490)")

# ---------- 3. unseen days ----------
N = 20_000
seeds = range(401, 401 + N)
ours = [L.play_day(sub, d)[0] for d in seeds]
his = [angkol_day(d, 20_000_000 + d) for d in seeds]
print(f"\n{N} unseen days: ours {st.mean(ours):.3f} +- {st.pstdev(ours):.2f}/day,"
      f" angkol clone {st.mean(his):.3f} -> ratio {st.mean(ours) / st.mean(his):.3f}")
print("our catch distribution:", sorted(Counter(ours).items()))
for size in (20, 50, 100, 400):
    ratios = [st.mean(ours[i:i + size]) / max(st.mean(his[i:i + size]), 1e-9)
              for i in range(0, N - size + 1, size)]
    below = sum(x < 1.5 for x in ratios)
    print(f"batches of {size:>3} days: min ratio {min(ratios):.3f}, "
          f"5th pct {sorted(ratios)[len(ratios) // 20]:.3f}, below 1.5x: {below}/{len(ratios)}")
