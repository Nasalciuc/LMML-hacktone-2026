"""In-process evaluation: imports an agent file, plays maps x 5 games, worst game per map.
usage: pypy3 evalx.py agent.py NMAPS SEED JOBS [out.json] [KEY=VAL ...]
KEY=VAL pairs override entries of the agent's PR dict."""
import importlib.util
import json
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, '.')
from arena import Arena

AGENT = None
OVR = {}


def load(path):
    spec = importlib.util.spec_from_file_location('agentmod', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if OVR and hasattr(m, 'PR'):
        m.PR.update(OVR)
    return m


def run(job):
    group, seed = job
    mod = load(AGENT)
    out = []
    with Arena() as a:
        for game in range(5):
            s = a.reset(group, seed, game)
            mem = {}
            think = 0.0
            dmax = 0.0
            illegal = 0
            first = {}
            while not s['game_done']:
                o = s['observation']
                for c, x in enumerate(o['claims']):
                    if x != -1 and c not in first:
                        first[c] = (o['round'], 0 if x == o['seat'] else 1)
                t0 = time.perf_counter()
                act = mod.agent(o, mem)
                dt = time.perf_counter() - t0
                think += dt
                dmax = max(dmax, dt)
                if act not in o['legal_actions']:
                    illegal += 1
                    act = {'op': 'wait'}
                s = a.step(act)
            sc = s['result']['scores']
            out.append(dict(group=group, seed=seed, game=game, hp=s['result']['half_points'],
                            me=sc[0], opp=sc[1], think=round(think, 3), illegal=illegal,
                            exp=mem.get('exp', 0), guard=mem.get('guard', 0), dmax=round(dmax, 3),
                            claims={str(k): v for k, v in first.items()}))
    return out


def main():
    global AGENT, OVR
    AGENT = sys.argv[1]
    n, seed, jobs = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    outp = sys.argv[5] if len(sys.argv) > 5 and '=' not in sys.argv[5] else None
    for kv in sys.argv[5:]:
        if '=' in kv:
            k, v = kv.split('=', 1)
            OVR[k] = float(v) if '.' in v or 'e' in v else int(v)
    rng = random.Random(seed)
    work = [(i % 8, rng.randrange(2 ** 32)) for i in range(n)]
    t0 = time.time()
    res = []
    with Pool(jobs, initializer=_init, initargs=(AGENT, OVR)) as p:
        for r in p.imap_unordered(run, work):
            res.extend(r)
    bymap = {}
    for r in res:
        bymap.setdefault((r['group'], r['seed']), []).append(r['hp'])
    worst = [min(v) for v in bymap.values()]
    score = sum(worst) / (2.0 * len(worst))
    won = sum(r['hp'] == 2 for r in res)
    draws = sum(r['hp'] == 1 for r in res)
    pg = {}
    for (g, s), v in bymap.items():
        pg.setdefault(g, []).append(min(v))
    print('SCORE %.3f  games won %d/%d (%.3f) draws %d  me %.1f opp %.1f  slowest %.2fs illegal %d wall %.0fs %s' % (
        score, won, len(res), won / len(res), draws, sum(r['me'] for r in res) / len(res),
        sum(r['opp'] for r in res) / len(res), max(r['think'] for r in res), sum(r['illegal'] for r in res),
        time.time() - t0, ' '.join('%s=%s' % kv for kv in OVR.items())))
    print('  by group ' + ' '.join('%d:%.2f' % (g, sum(v) / (2.0 * len(v))) for g, v in sorted(pg.items())))
    if outp:
        with open(outp, 'w') as f:
            json.dump(res, f)


def _init(a, o):
    global AGENT, OVR
    AGENT = a
    OVR = o


if __name__ == '__main__':
    main()
