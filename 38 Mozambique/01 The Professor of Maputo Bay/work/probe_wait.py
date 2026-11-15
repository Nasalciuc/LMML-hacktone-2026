"""Play 'wait' every round and log when the Professor claims each contract.
usage: python3 probe_wait.py out.jsonl NMAPS SEED JOBS"""
import json
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, '.')
from arena import Arena


def run(job):
    group, seed = job
    out = []
    with Arena() as a:
        for game in range(5):
            s = a.reset(group, seed, game)
            obs0 = s['observation']
            log = []
            while not s['game_done']:
                o = s['observation']
                log.append(dict(r=o['round'], claims=o['claims'], opp=o['opponent_score'],
                                eb=o['enemy_boats'], stocks=o['stocks']))
                s = a.step({'op': 'wait'})
            out.append(dict(group=group, seed=seed, game=game, map=obs0['map'], seat=obs0['seat'],
                            home=obs0['own']['home'], log=log, result=s['result']))
    return out


if __name__ == '__main__':
    path, n, seed, jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = random.Random(seed)
    work = [(i % 8, rng.randrange(2 ** 32)) for i in range(n)]
    with Pool(jobs) as p, open(path, 'w') as f:
        for res in p.imap_unordered(run, work):
            for g in res:
                f.write(json.dumps(g) + '\n')
