"""Compact summary of wait400.jsonl -> prof_sum.json: per game map, prof home, claim rounds, score."""
import json
out = []
for line in open('wait400.jsonl'):
    g = json.loads(line)
    cl = {}
    for e in g['log']:
        for c, x in enumerate(e['claims']):
            if x != -1 and c not in cl:
                cl[c] = e['r']          # observed at start of round r -> delivered in round r-1
    m = g['map']
    out.append(dict(group=g['group'], seed=g['seed'], game=g['game'], seat=g['seat'], home=g['home'],
                    phome=m['homes'][1 - g['seat']], roles=m['roles'], routes=m['routes'],
                    contracts=m['contracts'], claims=[cl.get(c, -1) for c in range(8)],
                    score=g['result']['scores'][1]))
json.dump(out, open('prof_sum.json', 'w'))
print(len(out))
