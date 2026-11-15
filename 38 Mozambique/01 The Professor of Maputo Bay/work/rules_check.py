"""Compare a hand-written rules model with the engine: legal actions and own state, random play.
usage: python3 rules_check.py NGAMES"""
import json
import random
import sys

sys.path.insert(0, '.')
from arena import Arena


def legal(o, stock_est):
    m = o['map']
    roles = m['roles']
    routes = m['routes']
    own = o['own']
    r = o['round']
    out = [{'op': 'wait'}]
    for b, bt in enumerate(own['boats']):
        if bt['eta']:
            continue
        p = bt['position']
        cargo = bt['cargo']
        free = 3 - sum(cargo)
        for d in range(len(routes)):
            if routes[p][d]:
                out.append({'op': 'sail', 'boat': b, 'destination': d})
        wh = own['warehouses'][p]
        for g in range(3):
            for q in range(1, min(wh[g], free) + 1):
                out.append({'op': 'load', 'boat': b, 'good': g, 'quantity': q})
        for g in range(3):
            for q in range(1, cargo[g] + 1):
                out.append({'op': 'unload', 'boat': b, 'good': g, 'quantity': q})
        if roles[p] in (0, 1):
            st = o['stocks'][p]
            for q in range(1, min(st, free) + 1):
                out.append({'op': 'gather', 'boat': b, 'good': roles[p], 'quantity': q})
        if roles[p] == 2:
            if not own['workshops'][p]:
                if wh[0] >= 2 and wh[1] >= 1:
                    out.append({'op': 'build_workshop', 'boat': b})
            elif own['jobs'][p] == 0 and wh[0] >= 2 and wh[1] >= 1:
                out.append({'op': 'craft', 'boat': b})
        for c, ct in enumerate(m['contracts']):
            if ct['destination'] == p and o['claims'][c] == -1 and r < ct['deadline'] and cargo[ct['good']] >= ct['quantity']:
                out.append({'op': 'deliver', 'boat': b, 'contract': c})
    hw = own['warehouses'][own['home']]
    if hw[0] >= 4 and hw[1] >= 2 and len(own['boats']) < 3:
        out.append({'op': 'build_boat'})
    return out


def key(a):
    return json.dumps(a, sort_keys=True)


def main():
    n = int(sys.argv[1])
    rng = random.Random(5)
    diffs = {}
    with Arena() as a:
        for gi in range(n):
            g = gi % 8
            s = a.reset(g, rng.randrange(2 ** 32), gi % 5)
            while not s['game_done']:
                o = s['observation']
                mine = set(key(x) for x in legal(o, None))
                eng = set(key(x) for x in o['legal_actions'])
                for x in mine - eng:
                    k = 'extra ' + json.loads(x)['op']
                    diffs[k] = diffs.get(k, 0) + 1
                    if diffs[k] < 3:
                        print(k, x, 'round', o['round'], json.dumps(o['own'])[:400])
                for x in eng - mine:
                    k = 'missing ' + json.loads(x)['op']
                    diffs[k] = diffs.get(k, 0) + 1
                    if diffs[k] < 3:
                        print(k, x, 'round', o['round'], json.dumps(o['own'])[:400])
                L = o['legal_actions']
                # bias towards productive actions
                pri = [x for x in L if x['op'] in ('deliver', 'build_workshop', 'craft', 'build_boat')]
                if pri and rng.random() < 0.8:
                    act = rng.choice(pri)
                else:
                    act = rng.choice(L)
                s = a.step(act)
    print(diffs)


if __name__ == '__main__':
    main()
