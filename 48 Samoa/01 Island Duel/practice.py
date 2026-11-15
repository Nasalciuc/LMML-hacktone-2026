"""Batch practice runner: many seeded games over all groups, in parallel.

python practice.py --agent agent.py --seeds 0-7 --workers 12 --out logs/run.json
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent / "task_samoa" / "island-duel-participant"
sys.path.insert(0, str(BUNDLE))


def _worker(args):
    agent_path, jobs, env_kw = args
    import os
    for k, v in env_kw.items():
        os.environ[k] = str(v)
    from client import ArenaClient
    from run_agent import load_agent

    choose, reset = load_agent(agent_path)
    out = []
    with ArenaClient() as env:
        for arena, seed in jobs:
            reset()
            r = env.reset(arena, seed)
            obs = r["observation"]
            seat = obs["seat"]
            prof = obs["opponent_profile"]
            t0 = time.time()
            n = 0
            while not r["done"]:
                a = choose(r["observation"])
                r = env.step(a)
                n += 1
            sc = r["scores"]
            out.append({
                "arena": arena, "seed": seed, "seat": seat, "profile": prof,
                "scores": sc, "winner": r["winner"],
                "win": r["winner"] == seat, "margin": sc[seat] - sc[1 - seat],
                "sec": round(time.time() - t0, 2),
            })
    return out


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summarize(games, label=""):
    n = len(games)
    w = sum(g["win"] for g in games)
    d = sum(g["winner"] is None for g in games)
    by_g = defaultdict(list)
    for g in games:
        by_g[g["arena"]].append(g)
    by_seat = defaultdict(list)
    by_prof = defaultdict(list)
    for g in games:
        by_seat[g["seat"]].append(g["win"])
        by_prof[g["profile"]].append(g["win"])
    lines = [f"{label} games={n} wins={w} ({w / max(n, 1):.3f}) draws={d}"
             f" mean_margin={sum(g['margin'] for g in games) / max(n, 1):.1f}"
             f" sec/game={sum(g['sec'] for g in games) / max(n, 1):.2f}"]
    lines.append("seat: " + "  ".join(f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(by_seat.items())))
    lines.append("profile: " + "  ".join(f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(by_prof.items())))
    cells = []
    for a in sorted(by_g):
        gs = by_g[a]
        cells.append(f"{a}:{sum(x['win'] for x in gs)}/{len(gs)}")
    lines.append("groups: " + " ".join(cells))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--arenas", default="0-31")
    p.add_argument("--seeds", default="0-3")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out")
    p.add_argument("--env", default="", help="KEY=VAL,KEY=VAL passed to workers")
    p.add_argument("--params", default="", help="json file with agent parameters")
    p.add_argument("--seed_base", type=int, default=1000)
    a = p.parse_args()
    env_kw = dict(kv.split("=") for kv in a.env.split(",") if kv)
    if a.params:
        d = json.loads(Path(a.params).read_text())
        d = d.get("best", d) if isinstance(d, dict) and "history" in d else d
        env_kw["AGENT_PARAMS"] = json.dumps(d)
    jobs = [(ar, a.seed_base + s) for s in parse_range(a.seeds) for ar in parse_range(a.arenas)]
    chunks = [jobs[i::a.workers] for i in range(a.workers)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(_worker, [(str(Path(a.agent).resolve()), c, env_kw) for c in chunks if c])
    games = sorted((g for r in res for g in r), key=lambda g: (g["arena"], g["seed"]))
    print(summarize(games, Path(a.agent).name + " " + a.env + " " + a.params))
    print(f"wall {time.time() - t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"agent": a.agent, "env": a.env, "games": games}) + "\n")


if __name__ == "__main__":
    main()
