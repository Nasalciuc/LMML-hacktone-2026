"""Offline pipeline test without the engine: our agent alone on real map layouts (the local rules copy is the
environment, the Professor never claims anything). Reports when we deliver the k-th manufactured contract.
usage: pypy3 solo.py agent.py maps.json NMAPS [KEY=VAL ...]   (maps.json = prof_sum2.json)"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
maps = json.load(open(sys.argv[2]))
n = int(sys.argv[3])
mode = 'choose'
for kv in sys.argv[4:]:
    k, v = kv.split('=')
    if k == 'mode':
        mode = v
    else:
        m.PR[k] = float(v) if '.' in v else int(v)
seen = set(); sel = []
for g in maps:
    key = (g['group'], g['seed'])
    if key not in seen:
        seen.add(key); sel.append(g)
    if len(sel) >= n:
        break
kth = [[] for _ in range(4)]; raws = []; tot = []; rolls = 0
for g in sel:
    nsite = len(g['roles'])
    obs = dict(round=0, seat=g['seat'], claims=[-1] * 8, stocks=[4 if g['roles'][s] in (0, 1) else None for s in range(nsite)],
               map=dict(roles=g['roles'], routes=g['routes'], contracts=g['contracts'],
                        homes=[g['home'], g['phome']] if g['seat'] == 0 else [g['phome'], g['home']]),
               own=dict(home=g['home'], score=0, boats=[dict(position=g['home'], destination=-1, eta=0, cargo=[0, 0, 0]) for _ in range(2)],
                        warehouses=[[5, 3, 0] if s == g['home'] else [0, 0, 0] for s in range(nsite)],
                        workshops=[False] * nsite, jobs=[0] * nsite))
    S = m.Static(obs)
    mem = dict(seen={})
    best = None
    for M in S.msites:
        pl = m.Planner(S, M, 2, 0); pl.nroll = 0
        L = pl.encode(obs, mem)
        G = [m.NOGOAL, m.NOGOAL]
        dels = []
        for r in range(80):
            pl = m.Planner(S, M, 2, r); pl.nroll = 0
            if mode == 'choose':
                v, a, G = pl.choose(L, G)
            else:
                G = list(G); a = pl.policy(L, G)
            rolls += pl.nroll
            if a[0] == m.DELIVER:
                dels.append((a[2], r))
            L = pl.apply(L, a)
        pts = sum(S.cpts[c] for c, r in dels)
        if best is None or pts > best[0]:
            best = (pts, dels)
    pts, dels = best
    mm = sorted(r for c, r in dels if S.cgood[c] == 2)
    for k in range(4):
        kth[k].append(mm[k] if k < len(mm) else 99)
    raws.append(sum(1 for c, r in dels if S.cgood[c] != 2)); tot.append(pts)
def q(v, f):
    v = sorted(v); return v[int(f * (len(v) - 1))]
print('%s %s maps %d  points avg %.1f  raws avg %.2f  rollouts/game %d' % (sys.argv[1], ' '.join(sys.argv[4:]), len(sel), sum(tot) / len(tot), sum(raws) / len(raws), rolls // (2 * len(sel))))
for k in range(4):
    print('  manuf #%d delivered: p10 %d p50 %d p90 %d  never %d%%' % (k + 1, q(kth[k], .1), q(kth[k], .5), q(kth[k], .9), 100 * sum(1 for x in kth[k] if x == 99) // len(sel)))
