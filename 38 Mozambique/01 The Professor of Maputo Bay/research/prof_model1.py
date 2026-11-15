"""How predictable are the Professor's claim rounds from map geometry? (unopposed games)"""
import json, collections, statistics as st
G = json.load(open('prof_sum.json'))
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
rows = []
for g in G:
    d = dists(g['routes']); h = g['phome']; roles = g['roles']
    ms = [s for s in range(10) if roles[s] == 2]
    g['d'] = d
    for c, ct in enumerate(g['contracts']):
        t = g['claims'][c]
        if ct['good'] == 2:
            via = min(d[h][m] + d[m][ct['destination']] for m in ms)
            rows.append(('M', c, t, via, d[h][ct['destination']], ct['deadline']))
        else:
            rows.append(('R', c, t, d[h][ct['destination']], 0, ct['deadline']))
# raw: claim round minus distance from prof home
for c in range(4):
    v = [r[2] - r[3] for r in rows if r[1] == c and r[2] >= 0]
    miss = sum(1 for r in rows if r[1] == c and r[2] < 0)
    v.sort(); n = len(v)
    print('raw slot', c, 'claimed', n, 'missed', miss, 't-dist: p5 %d p25 %d p50 %d p75 %d p95 %d' % (v[n // 20], v[n // 4], v[n // 2], v[3 * n // 4], v[19 * n // 20]))
for c in range(4, 8):
    v = [r[2] - r[3] for r in rows if r[1] == c and r[2] >= 0]
    miss = sum(1 for r in rows if r[1] == c and r[2] < 0)
    v.sort(); n = len(v)
    print('man slot', c, 'claimed', n, 'missed', miss, 't-via: min %d p5 %d p25 %d p50 %d p75 %d p95 %d' % (v[0], v[n // 20], v[n // 4], v[n // 2], v[3 * n // 4], v[19 * n // 20]))
# k-th manufactured claim (in time order) regardless of slot
kth = collections.defaultdict(list)
for g in G:
    ts = sorted(t for c, t in enumerate(g['claims']) if c >= 4 and t >= 0)
    for k, t in enumerate(ts):
        kth[k].append(t)
    g['nman'] = len(ts)
for k in range(4):
    v = sorted(kth[k]); n = len(v)
    print('k-th manuf claim', k, 'n', n, 'min %d p5 %d p10 %d p25 %d p50 %d p75 %d' % (v[0], v[n // 20], v[n // 10], v[n // 4], v[n // 2], v[3 * n // 4]))
# missed contract rate vs via distance for manuf
for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 99)):
    for c in range(4, 8):
        sel = [r for r in rows if r[1] == c and lo <= r[3] < hi]
        if sel:
            print('man slot', c, 'via in [%d,%d)' % (lo, hi), 'n', len(sel), 'claimed %.0f%%' % (100 * sum(r[2] >= 0 for r in sel) / len(sel)),
                  'median t', sorted(r[2] for r in sel if r[2] >= 0)[len([r for r in sel if r[2] >= 0]) // 2] if any(r[2] >= 0 for r in sel) else None)
# style signature: number of raws claimed before round 20, first manuf round
sig = collections.Counter()
for g in G:
    nr = sum(1 for c in range(4) if 0 <= g['claims'][c] < 20)
    fm = min([t for c, t in enumerate(g['claims']) if c >= 4 and t >= 0] or [99])
    sig[(nr, min(fm // 8, 6))] += 1
print('signature (raws before r20, first manuf round // 8):')
for k, v in sorted(sig.items()):
    print('  ', k, v)
