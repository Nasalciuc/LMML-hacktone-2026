import json
import random
import sys

sys.path.insert(0, '.')
from arena import Arena

rng = random.Random(1)
with Arena() as a:
    s = a.reset(2, 12345, 0)
    o = s['observation']
    print(json.dumps(o['map']))
    while not s['game_done']:
        o = s['observation']
        pos = [(b['position'], b['destination'], b['eta']) for b in o['own']['boats']]
        print(o['round'], 'pos', pos, 'vis', [i for i, v in enumerate(o['visible']) if v],
              'eb', o['enemy_boats'], 'st', o['stocks'], 'cl', o['claims'], 'opp', o['opponent_score'])
        sails = [x for x in o['legal_actions'] if x['op'] == 'sail']
        act = rng.choice(sails) if sails and rng.random() < 0.7 else {'op': 'wait'}
        s = a.step(act)
    print(s['result'])
