"""Unopposed Professor games (we wait every round); writes only a compact summary.
usage: python3 prof_collect.py out.json NMAPS SEED JOBS"""
import json, random, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from arena import Arena


def run(job):
    group, seed = job
    out = []
    with Arena() as a:
        for game in range(5):
            s = a.reset(group, seed, game)
            o0 = s['observation']
            cl = {}
            while not s['game_done']:
                o = s['observation']
                for c, x in enumerate(o['claims']):
                    if x != -1 and c not in cl:
                        cl[c] = o['round']
                s = a.step({'op': 'wait'})
            m = o0['map']
            out.append(dict(group=group, seed=seed, game=game, seat=o0['seat'], home=o0['own']['home'],
                            phome=m['homes'][1 - o0['seat']], roles=m['roles'], routes=m['routes'],
                            contracts=m['contracts'], claims=[cl.get(c, -1) for c in range(8)],
                            score=s['result']['scores'][1]))
    return out


if __name__ == '__main__':
    path, n, seed, jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = random.Random(seed)
    work = [(i % 8, rng.randrange(2 ** 32)) for i in range(n)]
    res = []
    with Pool(jobs) as p:
        for r in p.imap_unordered(run, work):
            res.extend(r)
    json.dump(res, open(path, 'w'))
    print(len(res))
