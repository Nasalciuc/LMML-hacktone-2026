"""Does the Professor react to what we do? Same game, different scripted openings."""
import json
import sys

sys.path.insert(0, '.')
from arena import Arena


def claims_of(a, group, seed, game, script):
    s = a.reset(group, seed, game)
    cl = [None] * 8
    while not s['game_done']:
        o = s['observation']
        for c, x in enumerate(o['claims']):
            if x != -1 and cl[c] is None:
                cl[c] = o['round']
        act = script(o)
        if act not in o['legal_actions']:
            act = {'op': 'wait'}
        s = a.step(act)
    return cl, s['result']['scores'][1]


def wait(o):
    return {'op': 'wait'}


def boat(o):
    return {'op': 'build_boat'} if o['round'] == 0 else {'op': 'wait'}


def dump_home(o):
    # load all home goods on boats (home warehouse empty), stay home
    r = o['round']
    seq = [{'op': 'load', 'boat': 0, 'good': 0, 'quantity': 3}, {'op': 'load', 'boat': 1, 'good': 0, 'quantity': 2},
           {'op': 'load', 'boat': 1, 'good': 1, 'quantity': 1}]
    return seq[r] if r < len(seq) else {'op': 'wait'}


def sail_away(o):
    if o['round'] < 2:
        b = o['round']
        for x in o['legal_actions']:
            if x['op'] == 'sail' and x['boat'] == b:
                return x
    return {'op': 'wait'}


if __name__ == '__main__':
    cases = [(0, 767302775), (3, 569109051), (4, 207977398), (6, 111), (1, 3582206753)]
    with Arena() as a:
        for g, sd in cases:
            for game in range(2):
                for name, sc in [('wait', wait), ('boat', boat), ('dump', dump_home), ('sail', sail_away)]:
                    cl, ps = claims_of(a, g, sd, game, sc)
                    print(g, sd, game, name, ps, cl)
