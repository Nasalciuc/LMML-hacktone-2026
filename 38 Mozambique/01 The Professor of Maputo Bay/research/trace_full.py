"""Full action trace of one game. usage: pypy3 trace_full.py agent.py group seed game [KEY=VAL ...]"""
import json, sys, importlib.util
sys.path.insert(0, '.')
from arena import Arena
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for kv in sys.argv[5:]:
    k, v = kv.split('=')
    m.PR[k] = float(v) if '.' in v else int(v)
g, sd, game = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with Arena() as a:
    s = a.reset(g, sd, game); mem = {}
    o = s['observation']
    print('roles', o['map']['roles'], 'homes', o['map']['homes'], 'seat', o['seat'])
    print('routes', [(i, j, o['map']['routes'][i][j]) for i in range(10) for j in range(i + 1, 10) if o['map']['routes'][i][j]])
    for c, ct in enumerate(o['map']['contracts']):
        print('contract', c, ct)
    prev = list(o['claims'])
    while not s['game_done']:
        o = s['observation']
        ev = ['c%d->%s' % (c, 'me' if x == o['seat'] else 'PROF') for c, x in enumerate(o['claims']) if x != prev[c]]
        prev = list(o['claims'])
        act = m.agent(o, mem)
        own = o['own']
        boats = ' '.join('b%d@%s%s%s' % (i, b['position'], ('>%d(%d)' % (b['destination'], b['eta'])) if b['eta'] else '', b['cargo']) for i, b in enumerate(own['boats']))
        wh = {i: w for i, w in enumerate(own['warehouses']) if any(w)}
        a2 = dict(act); op = a2.pop('op')
        print('r%2d %-15s %-40s | %s wh%s jobs%s %d:%d %s' % (o['round'], op, json.dumps(a2), boats, wh,
              [j for j in own['jobs'] if j], own['score'], o['opponent_score'], ' '.join(ev)))
        s = a.step(act)
    print('result', s['result'])
