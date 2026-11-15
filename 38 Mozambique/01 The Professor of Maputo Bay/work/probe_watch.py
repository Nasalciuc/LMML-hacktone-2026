"""Watch the Professor: park our boats on every site over several runs of the same game
and merge what was seen. usage: python3 probe_watch.py group seed game"""
import json
import sys

sys.path.insert(0, '.')
from arena import Arena


def bfs_next(routes, a, b):
    n = len(routes)
    best = {a: (0, None)}
    todo = [a]
    # dijkstra by time
    dist = {a: 0}
    first = {a: None}
    seen = set()
    while True:
        cur = None
        for v in dist:
            if v not in seen and (cur is None or dist[v] < dist[cur]):
                cur = v
        if cur is None:
            break
        seen.add(cur)
        for w in range(n):
            if routes[cur][w] and w not in seen:
                d = dist[cur] + routes[cur][w]
                if w not in dist or d < dist[w]:
                    dist[w] = d
                    first[w] = w if cur == a else first[cur]
    return first.get(b)


def watch(group, seed, game):
    runs = []
    with Arena() as a:
        s = a.reset(group, seed, game)
        o = s['observation']
        routes = o['map']['routes']
        home = o['own']['home']
        sites = [x for x in range(len(routes)) if x != home]
        parts = [sites[i:i + 3] for i in range(0, len(sites), 3)]
        for part in parts:
            s = a.reset(group, seed, game)
            log = []
            while not s['game_done']:
                o = s['observation']
                log.append(dict(r=o['round'], eb=o['enemy_boats'], vis=o['visible'], st=o['stocks'],
                                cl=o['claims'], opp=o['opponent_score']))
                legal = o['legal_actions']
                act = {'op': 'wait'}
                if {'op': 'build_boat'} in legal:
                    act = {'op': 'build_boat'}
                else:
                    for b, bt in enumerate(o['own']['boats']):
                        if b >= len(part) or bt['eta']:
                            continue
                        if bt['position'] != part[b]:
                            nx = bfs_next(routes, bt['position'], part[b])
                            cand = {'op': 'sail', 'boat': b, 'destination': nx}
                            if cand in legal:
                                act = cand
                                break
                s = a.step(act)
            runs.append(dict(part=part, log=log, result=s['result']))
    return dict(group=group, seed=seed, game=game, map=o['map'], home=home, runs=runs)


if __name__ == '__main__':
    g, sd, gm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    res = watch(g, sd, gm)
    # merged timeline
    T = {}
    for run in res['runs']:
        for e in run['log']:
            for b in e['eb']:
                T.setdefault(e['r'], set()).add((b['boat'], b['position'], b['destination'], b['eta']))
    print(json.dumps(res['map']))
    print('home', res['home'], 'results', [r['result']['scores'] for r in res['runs']])
    print('claims per run equal:', len(set(json.dumps(r['log'][-1]['cl']) for r in res['runs'])) == 1)
    st = {}
    for run in res['runs']:
        for e in run['log']:
            for i, v in enumerate(e['st']):
                if v is not None and e['vis'][i]:
                    st.setdefault(e['r'], {})[i] = v
    prev_cl = None
    for r in range(80):
        cl = res['runs'][0]['log'][r]['cl'] if r < len(res['runs'][0]['log']) else None
        print(r, sorted(T.get(r, [])), 'st', st.get(r), 'cl', cl if cl != prev_cl else '', res['runs'][0]['log'][r]['opp'])
        prev_cl = cl
