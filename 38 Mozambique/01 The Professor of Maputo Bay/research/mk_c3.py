"""c3 = c2 + scripted manufacturing-first opening (PR['open']=1).
Haul home timber+fibre straight to the workshop site, build, start the first batch; then hand over to the beam search."""
src = open('c2.py').read()

def rep(a, b):
    global src
    assert a in src, a
    src = src.replace(a, b, 1)

rep("    'boat3': -1,", "    'boat3': -1,\n    'open': 0,          # 1 = scripted manufacturing-first opening\n")

OPEN = '''
def opening(obs, memory):
    """One action of the manufacturing-first opening, or None when finished."""
    S = memory['S']
    M = memory['M']
    own = obs['own']
    legal = obs['legal_actions']
    boats = own['boats']
    home = S.home
    wh = own['warehouses']
    if own['workshops'][M] and own['jobs'][M] > 0:
        return None                         # first batch is running: done
    if obs['round'] > 30:
        return None
    def ok(a):
        return a if a in legal else None
    # 1) at the workshop site: unload, build, craft
    for b, bt in enumerate(boats):
        if bt['eta'] == 0 and bt['position'] == M:
            for g in (T, F):
                if bt['cargo'][g]:
                    a = ok({'op': 'unload', 'boat': b, 'good': g, 'quantity': bt['cargo'][g]})
                    if a:
                        return a
    a = None
    for b, bt in enumerate(boats):
        if bt['eta'] == 0 and bt['position'] == M:
            if not own['workshops'][M]:
                a = ok({'op': 'build_workshop', 'boat': b})
            else:
                a = ok({'op': 'craft', 'boat': b})
            if a:
                return a
    # 2) at home: load what the workshop still needs (4 timber + 2 fibre in total)
    carried = [sum(bt['cargo'][g] for bt in boats) + wh[M][g] for g in (T, F)]
    need = [max(0, 4 - carried[0]), max(0, 2 - carried[1])]
    if own['workshops'][M]:
        need = [max(0, 2 - carried[0]), max(0, 1 - carried[1])]
    for b, bt in enumerate(boats):
        if bt['eta'] == 0 and bt['position'] == home:
            free = CAP - sum(bt['cargo'])
            for g in (T, F):
                q = min(free, need[g], wh[home][g])
                if q > 0:
                    a = ok({'op': 'load', 'boat': b, 'good': g, 'quantity': q})
                    if a:
                        return a
    # 3) loaded boats sail one hop towards the workshop
    for b, bt in enumerate(boats):
        if bt['eta'] == 0 and bt['position'] != M and (bt['cargo'][T] or bt['cargo'][F]):
            nx = S.nxt[bt['position']][M]
            a = ok({'op': 'sail', 'boat': b, 'destination': nx})
            if a:
                return a
    if any(bt['eta'] for bt in boats):
        return None if sum(need) == 0 and not any(bt['cargo'][T] or bt['cargo'][F] for bt in boats) else {'op': 'wait'}
    return None
'''
rep("def est_stock(mem, site, rnd):", OPEN.lstrip('\n') + "\n\ndef est_stock(mem, site, rnd):")

rep("    pl = Planner(S, memory['M'], nb, rnd)\n",
    "    if PR.get('open') and not memory.get('open_done'):\n"
    "        oa = opening(obs, memory)\n"
    "        if oa is None:\n"
    "            memory['open_done'] = True\n"
    "        else:\n"
    "            memory['plan'] = None\n"
    "            memory['used'] += time.perf_counter() - t0\n"
    "            return oa\n"
    "    pl = Planner(S, memory['M'], nb, rnd)\n")
open('c3.py', 'w').write(src)
print('c3 ok')
