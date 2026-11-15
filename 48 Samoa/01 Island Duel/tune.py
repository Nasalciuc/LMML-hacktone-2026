"""Tune agent parameters against the practice opponent with a small CMA-ES.

python tune.py --agent agent2.py --params t_neu,t_cap,look --seeds 0-15 --gens 30
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent.parent / "task_samoa" / "island-duel-participant"
sys.path.insert(0, str(BUNDLE))

_W = {}


def _init(agent_path):
    from client import ArenaClient
    from run_agent import load_agent
    import importlib
    _W["env"] = ArenaClient()
    choose, reset = load_agent(agent_path)
    _W["choose"], _W["reset"] = choose, reset
    _W["mod"] = sys.modules["participant_agent"]


def _play(task):
    params, jobs = task
    _W["mod"].set_params(params)
    env, choose, reset = _W["env"], _W["choose"], _W["reset"]
    out = []
    for arena, seed in jobs:
        reset()
        r = env.reset(arena, seed)
        seat = r["observation"]["seat"]
        while not r["done"]:
            r = env.step(choose(r["observation"]))
        sc = r["scores"]
        out.append((arena, seed, seat, sc[seat] - sc[1 - seat]))
    return out


class Runner:
    def __init__(self, agent, workers):
        self.pool = mp.Pool(workers, initializer=_init, initargs=(str(Path(agent).resolve()),))
        self.workers = workers

    def run(self, params_list, jobs):
        tasks = []
        idx = []
        chunk = max(1, len(jobs) // (self.workers * 2))
        for pi, p in enumerate(params_list):
            for k in range(0, len(jobs), chunk):
                tasks.append((p, jobs[k:k + chunk]))
                idx.append(pi)
        res = self.pool.map(_play, tasks)
        out = [[] for _ in params_list]
        for pi, r in zip(idx, res):
            out[pi] += r
        return out


def stats(games):
    m = np.array([g[3] for g in games], dtype=float)
    return {
        "n": len(m), "win": float((m > 0).mean()), "margin": float(m.mean()),
        "obj": float(np.tanh(m / 30.0).mean()),
    }


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent2.py")
    ap.add_argument("--params", required=True, help="comma list of parameter names to tune")
    ap.add_argument("--init", default="", help="json file or string with starting params")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seeds", default="0-15")
    ap.add_argument("--seed_base", type=int, default=20000)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="logs/tune.json")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.agent).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("agentdefaults", Path(a.agent).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = dict(mod.DEFAULT)
    if a.init:
        txt = Path(a.init).read_text() if Path(a.init).exists() else a.init
        base.update(json.loads(txt))
    names = a.params.split(",")
    scale = np.array([max(abs(base[n]), 1.0) for n in names])
    x0 = np.array([base[n] for n in names], dtype=float)

    jobs = [(ar, a.seed_base + s) for s in parse_range(a.seeds) for ar in range(32)]
    runner = Runner(a.agent, a.workers)

    def to_params(z):
        p = dict(base)
        for n, v in zip(names, z):
            p[n] = float(v)
        return p

    t0 = time.time()
    base_res = stats(runner.run([to_params(x0)], jobs)[0])
    print("base", base_res, flush=True)
    history = [{"params": to_params(x0), **base_res}]

    # simple CMA-ES in normalised coordinates
    n = len(names)
    lam = a.pop
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1 / (wts ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    mean = np.zeros(n)
    sigma = a.sigma
    Cm = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    best = (base_res["obj"], to_params(x0), base_res)
    rng = np.random.default_rng(0)
    for g in range(a.gens):
        D2, B = np.linalg.eigh(Cm)
        D = np.sqrt(np.maximum(D2, 1e-12))
        zs = rng.standard_normal((lam, n))
        ys = zs @ (B * D).T
        xs = mean + sigma * ys
        plist = [to_params(x0 + scale * x) for x in xs]
        res = [stats(r) for r in runner.run(plist, jobs)]
        objs = np.array([r["obj"] for r in res])
        order = np.argsort(-objs)
        for i in order[:1]:
            if res[i]["obj"] > best[0]:
                best = (res[i]["obj"], plist[i], res[i])
        for p, r in zip(plist, res):
            history.append({"params": p, **r})
        ysel = ys[order[:mu]]
        yw = wts @ ysel
        mean = mean + sigma * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        Cm = (1 - c1 - cmu) * Cm + c1 * np.outer(pc, pc) + cmu * (ysel.T * wts) @ ysel
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        mp_ = to_params(x0 + scale * mean)
        print(f"gen {g} {time.time() - t0:.0f}s best_gen win={res[order[0]]['win']:.3f} "
              f"m={res[order[0]]['margin']:.1f} obj={objs[order[0]]:.3f} | mean_obj={objs.mean():.3f} "
              f"sigma={sigma:.3f} | mean " + " ".join(f"{k}={mp_[k]:.3f}" for k in names), flush=True)
        Path(a.out).write_text(json.dumps({"best": best[1], "best_stats": best[2],
                                            "mean": mp_, "history": history}, indent=1))
    # final check of the distribution mean
    fin = stats(runner.run([mp_], jobs)[0])
    print("final mean", fin, json.dumps({k: round(mp_[k], 4) for k in names}))
    Path(a.out).write_text(json.dumps({"best": best[1], "best_stats": best[2],
                                        "mean": mp_, "mean_stats": fin, "history": history}, indent=1))


if __name__ == "__main__":
    main()
