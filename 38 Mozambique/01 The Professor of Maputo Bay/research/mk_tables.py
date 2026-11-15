"""Survival tables of the Professor's delivery round per contract slot and distance, from unopposed games.
S[t] = P(Professor has not delivered the contract before round t) (+ half of the same-round mass), in permille.
Manufactured slots 4-7 are indexed by via = min over manufacturing sites m of d(prof home, m) + d(m, destination),
raw slots 0-3 by d(prof home, destination).
usage: pypy3 mk_tables.py prof_sum.json prof_sum2.json > tables.txt"""
import json, sys, collections
INF = 99


def dists(routes):
    n = len(routes)
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
        for j in range(n):
            if routes[i][j]:
                d[i][j] = routes[i][j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


G = []
for p in sys.argv[1:]:
    G += json.load(open(p))
MLO, MHI, RLO, RHI = 2, 13, 1, 9
cells = collections.defaultdict(list)
for g in G:
    d = dists(g['routes']); h = g['phome']
    ms = [s for s in range(len(g['roles'])) if g['roles'][s] == 2]
    for c, ct in enumerate(g['contracts']):
        t = g['claims'][c]
        dr = t - 1 if t >= 0 else 999
        if ct['good'] == 2:
            f = min(d[h][m] + d[m][ct['destination']] for m in ms)
            f = max(MLO, min(MHI, f))
        else:
            f = max(RLO, min(RHI, d[h][ct['destination']]))
        cells[(c, f)].append(dr)


def surv(v):
    n = float(len(v))
    out = []
    for t in range(81):
        gt = sum(1 for x in v if x > t); eq = sum(1 for x in v if x == t)
        out.append(int(round(1000 * (gt + 0.5 * eq) / n)))
    return out


tab = {}
info = []
for c in range(8):
    lo, hi = (MLO, MHI) if c >= 4 else (RLO, RHI)
    row = []
    for f in range(lo, hi + 1):
        v = list(cells.get((c, f), []))
        w = 1
        while len(v) < 200 and w < 6:          # widen sparse cells with neighbours
            v = []
            for ff in range(f - w, f + w + 1):
                v += cells.get((c, ff), [])
            w += 1
        row.append(surv(v))
        info.append((c, f, len(cells.get((c, f), [])), len(v), row[-1][20], row[-1][35], row[-1][50], row[-1][80]))
    tab[c] = row
print('# games %d ; manufactured via range %d..%d ; raw distance range %d..%d' % (len(G), MLO, MHI, RLO, RHI))
print('SURV = ' + json.dumps([tab[c] for c in range(8)], separators=(',', ':')))
sys.stderr.write('slot feat n n_used S20 S35 S50 S80\n')
for x in info:
    sys.stderr.write(' '.join(str(y) for y in x) + '\n')
