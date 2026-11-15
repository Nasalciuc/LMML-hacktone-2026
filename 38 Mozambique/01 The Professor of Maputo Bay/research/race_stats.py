"""From an evalx result file: when do we / the Professor claim the k-th manufactured contract, who gets what."""
import json, sys, collections
for path in sys.argv[1:]:
    res = json.load(open(path))
    mine = collections.defaultdict(list); prof = collections.defaultdict(list)
    split = collections.Counter(); out = collections.Counter()
    for r in res:
        cl = {int(c): v for c, v in r['claims'].items()}
        mm = sorted(v[0] for c, v in cl.items() if c >= 4 and v[1] == 0)
        pm = sorted(v[0] for c, v in cl.items() if c >= 4 and v[1] == 1)
        mr = sum(1 for c, v in cl.items() if c < 4 and v[1] == 0)
        pr = sum(1 for c, v in cl.items() if c < 4 and v[1] == 1)
        for k, t in enumerate(mm): mine[k].append(t)
        for k, t in enumerate(pm): prof[k].append(t)
        split[(len(mm), len(pm))] += 1
        out[(len(mm), mr, len(pm), pr, r['hp'])] += 1
    n = len(res)
    print(path, 'games', n)
    for k in range(4):
        for name, d in (('me', mine), ('prof', prof)):
            v = sorted(d[k])
            if v:
                print('  %s manuf #%d: in %.0f%% of games, round p10 %d p50 %d p90 %d' % (name, k + 1, 100 * len(v) / n, v[len(v) // 10], v[len(v) // 2], v[9 * len(v) // 10]))
    print('  manuf split (me,prof):', sorted(split.items(), key=lambda x: -x[1])[:8])
    print('  top outcomes (myM,myR,profM,profR,hp):', sorted(out.items(), key=lambda x: -x[1])[:12])
