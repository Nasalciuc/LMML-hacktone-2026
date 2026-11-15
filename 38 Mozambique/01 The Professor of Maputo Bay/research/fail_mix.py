"""Why does a map fail? For each result file: per-map worst game, and the margin distribution of the worst games."""
import json, sys, collections
for path in sys.argv[1:]:
    res = json.load(open(path))
    bymap = collections.defaultdict(list)
    for r in res:
        bymap[(r['group'], r['seed'])].append(r)
    worst = []
    for v in bymap.values():
        w = min(v, key=lambda x: (x['hp'], x['me'] - x['opp']))
        worst.append(w)
    hp = collections.Counter(w['hp'] for w in worst)
    marg = collections.Counter()
    for w in worst:
        if w['hp'] < 2:
            d = w['me'] - w['opp']
            marg['draw' if d == 0 else ('-1..-7' if d >= -7 else ('-8..-14' if d >= -14 else '<=-15'))] += 1
    nlose = [sum(x['hp'] == 0 for x in v) for v in bymap.values()]
    ndraw = [sum(x['hp'] == 1 for x in v) for v in bymap.values()]
    fail1 = sum(1 for a, b in zip(nlose, ndraw) if a + b == 1)
    print(path, 'maps', len(bymap), 'worst hp', dict(hp), 'worst-game margins', dict(marg),
          'maps failing by exactly one game', fail1,
          'draw-only maps', sum(1 for a, b in zip(nlose, ndraw) if a == 0 and b > 0))
