"""Mozambique - The Professor of Maputo Bay. Submission: this file only (stdlib, PyPy 3.10).

Beam search over our own actions (one action per round for the whole player) on a local
copy of the rules. Raw contracts are valued at half their points inside the search, which
pushes the plan towards manufactured contracts (12 points), the ones that decide the race
against the Professor. A hard time guard keeps a game well inside the 10 s limit.
The serve() loop at the bottom is the template's and must stay as it is.
"""
import json
import sys
import time

T, F, P = 0, 1, 2
CAP = 3
ROUNDS = 80
INF = 10 ** 6

# tunable parameters
PR = {
    'gamma': 0.99,      # per-round discount of future points
    'lam': 0.93,        # per cost-unit discount of materials away from the workshop
    'mu': 0.85,         # extra discount for materials sitting in a warehouse away from the workshop
    'kp': 0.85,         # value factor of an existing product
    'kb': 0.6,          # value factor of one batch worth of materials
    'kr': 0.8,          # value factor of carried raw goods matched to a raw contract
    'nu': 0.6,          # value factor of natural stock that could be gathered
    'unit': 0.25,       # per-unit bonus for useful materials
    'build': 3.0,       # value of having the workshop built
    'risk': 1.0,        # value factor when we arrive after the professor's estimated time
    'raw_eta': 3,       # professor slack for raw contracts
    'man_eta': 22,      # professor lead time for manufactured contracts
    'H': 14,            # search horizon
    'W': 24,            # beam width
    'msel': 1,          # choose workshop site by comparing opening searches
    'H0': 0,            # deeper opening search (0 = off)
    'W0': 20,
    'T0': 1.0,
    'budget': 6.0,      # seconds of search per game (soft, spread over the rounds)
    'ebud': 0,          # node expansions per game for the search (0 = wall-clock budget)
    'e0': 60000,        # expansion cap of each round-0 workshop-site search
    'hard': 8.0,        # seconds per game after which we stop searching (limit is 10)
    'rawv': 0.5,        # raw contracts count at this fraction of their points in the search
    'tieb': 99,         # from this round on raw contracts count fully while not ahead (99 = off)
    '_rawf': 0.5,       # (internal) raw factor in force for the current decision
}

# op codes
WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)


class Static(object):
    def __init__(self, obs):
        m = obs['map']
        self.routes = routes = m['routes']
        self.n = n = len(routes)
        self.roles = m['roles']
        self.contracts = m['contracts']
        self.nc = len(self.contracts)
        self.home = obs['own']['home']
        self.seat = obs['seat']
        homes = m.get('homes') or [self.home, self.home]
        self.ohome = homes[1 - self.seat] if len(homes) > 1 else self.home
        self.adj = [[w for w in range(n) if routes[v][w]] for v in range(n)]
        dist = [[INF] * n for _ in range(n)]
        hops = [[INF] * n for _ in range(n)]
        nxt = [[-1] * n for _ in range(n)]
        for s in range(n):
            best = {s: (0, 0)}
            first = {s: -1}
            done = set()
            while True:
                cur, cb = None, None
                for v, c in best.items():
                    if v not in done and (cb is None or c < cb):
                        cur, cb = v, c
                if cur is None:
                    break
                done.add(cur)
                for w in self.adj[cur]:
                    if w in done:
                        continue
                    cand = (cb[0] + routes[cur][w], cb[1] + 1)
                    if w not in best or cand < best[w]:
                        best[w] = cand
                        first[w] = w if cur == s else first[cur]
            for v, (ct, ch) in best.items():
                dist[s][v] = ct
                hops[s][v] = ch
                nxt[s][v] = first[v]
        self.dist, self.hops, self.nxt = dist, hops, nxt
        self.cost = [[dist[a][b] + hops[a][b] if a != b else 0 for b in range(n)] for a in range(n)]
        self.tsites = [s for s in range(n) if self.roles[s] == T]
        self.fsites = [s for s in range(n) if self.roles[s] == F]
        self.msites = [s for s in range(n) if self.roles[s] == 2]
        self.natural = [s for s in range(n) if self.roles[s] in (T, F)]
        self.cdest = [c['destination'] for c in self.contracts]
        self.cgood = [c['good'] for c in self.contracts]
        self.cqty = [c['quantity'] for c in self.contracts]
        self.cpts = [c['points'] for c in self.contracts]
        self.cdl = [c['deadline'] for c in self.contracts]
        self.manuf = [c for c in range(self.nc) if self.cgood[c] == P]
        self.raws = [c for c in range(self.nc) if self.cgood[c] != P]
        self.dest_contracts = [[c for c in range(self.nc) if self.cdest[c] == s] for s in range(n)]
        # professor's estimated earliest delivery per contract
        oh = self.ohome
        mo = min(self.msites, key=lambda s: self.cost[oh][s]) if self.msites else oh
        eta = []
        for c in range(self.nc):
            if self.cgood[c] == P:
                eta.append(PR['man_eta'] + self.cost[oh][mo] + self.cost[mo][self.cdest[c]])
            else:
                eta.append(self.cost[oh][self.cdest[c]] + 2 + PR['raw_eta'])
        self.oeta = eta


def pick_workshop(S):
    best, bs = None, None
    for s in S.msites:
        sc = 2.0 * S.cost[S.home][s]
        for c in S.manuf:
            sc += 0.4 * S.cost[s][S.cdest[c]]
        nt = min([S.cost[s][x] for x in S.tsites] or [9])
        nf = min([S.cost[s][x] for x in S.fsites] or [9])
        sc += 1.0 * nt + 0.6 * nf
        if bs is None or sc < bs:
            best, bs = s, sc
    return best


class Planner(object):
    def __init__(self, S, M, nb, r0):
        self.S = S
        self.M = M
        self.nb = nb
        self.r0 = r0
        n = S.n
        self.WH = 4 + 6 * nb
        self.JB = self.WH + 3 * n
        self.ST = self.JB + n
        self.size = self.ST + n
        g = PR['gamma']
        self.disc = [g ** max(0, t - r0) for t in range(0, 200)]
        gm = PR.get('gm', g)
        self.discm = [gm ** max(0, t - r0) for t in range(0, 200)]
        lam = PR['lam']
        self.lamp = [lam ** k for k in range(200)]

    # ------------------------------------------------------------ encoding
    def encode(self, obs, mem):
        S = self.S
        own = obs['own']
        L = [0] * self.size
        L[0] = obs['round']
        L[1] = 0.0
        wsm = 0
        for s, x in enumerate(own['workshops']):
            if x:
                wsm |= 1 << s
        L[2] = wsm
        cm = 0
        for c, x in enumerate(obs['claims']):
            if x != -1:
                cm |= 1 << c
        L[3] = cm
        for b, bt in enumerate(own['boats'][:self.nb]):
            i = 4 + 6 * b
            L[i] = bt['position']
            L[i + 1] = bt['destination']
            L[i + 2] = bt['eta']
            L[i + 3], L[i + 4], L[i + 5] = bt['cargo']
        WH = self.WH
        for s, w in enumerate(own['warehouses']):
            L[WH + 3 * s], L[WH + 3 * s + 1], L[WH + 3 * s + 2] = w
        for s, j in enumerate(own['jobs']):
            L[self.JB + s] = j
        for s in range(S.n):
            if S.roles[s] in (T, F):
                L[self.ST + s] = est_stock(mem, s, L[0])
        return L

    # ------------------------------------------------------------ dynamics
    def actions(self, L):
        S = self.S
        WH, JB, ST = self.WH, self.JB, self.ST
        r = L[0]
        acts = [(WAIT, 0, 0, 0)]
        wsm = L[2]
        closed = L[3]
        for b in range(self.nb):
            i = 4 + 6 * b
            if L[i + 2]:
                continue
            p = L[i]
            cT, cF, cP = L[i + 3], L[i + 4], L[i + 5]
            free = CAP - cT - cF - cP
            # deliveries
            for c in S.dest_contracts[p]:
                if not (closed >> c) & 1 and r < S.cdl[c] and L[i + 3 + S.cgood[c]] >= S.cqty[c]:
                    acts.append((DELIVER, b, c, 0))
            role = S.roles[p]
            w = WH + 3 * p
            if role == 2:
                if not (wsm >> p) & 1:
                    if L[w] >= 2 and L[w + 1] >= 1:
                        acts.append((BUILD, b, 0, 0))
                elif L[JB + p] == 0 and L[w] >= 2 and L[w + 1] >= 1:
                    acts.append((CRAFT, b, 0, 0))
                if cT:
                    acts.append((UNLOAD, b, T, cT))
                if cF:
                    acts.append((UNLOAD, b, F, cF))
                if L[w + 2] and free:
                    acts.append((LOAD, b, P, min(free, L[w + 2])))
            else:
                if free:
                    for g in (T, F):
                        x = L[w + g]
                        if x:
                            for q in range(1, min(x, free) + 1):
                                acts.append((LOAD, b, g, q))
                if (role == T or role == F) and free:
                    x = L[ST + p]
                    if x > 0:
                        for q in range(1, min(x, free) + 1):
                            acts.append((GATHER, b, role, q))
            for d in S.adj[p]:
                acts.append((SAIL, b, d, 0))
        return acts

    def apply(self, L, a):
        S = self.S
        L = L[:]
        op, b, x, q = a
        i = 4 + 6 * b
        WH, JB, ST = self.WH, self.JB, self.ST
        r = L[0]
        if op == SAIL:
            L[i + 1] = x
            L[i + 2] = S.routes[L[i]][x]
        elif op == LOAD:
            L[WH + 3 * L[i] + x] -= q
            L[i + 3 + x] += q
        elif op == UNLOAD:
            L[WH + 3 * L[i] + x] += q
            L[i + 3 + x] -= q
        elif op == GATHER:
            L[ST + L[i]] -= q
            L[i + 3 + x] += q
        elif op == BUILD:
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[2] |= 1 << L[i]
        elif op == CRAFT:
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[JB + L[i]] = 3
        elif op == DELIVER:
            c = x
            L[i + 3 + S.cgood[c]] -= S.cqty[c]
            L[3] |= 1 << c
            v = S.cpts[c] * (self.discm[r] if S.cgood[c] == P else self.disc[r] * PR['_rawf'])
            if r > S.oeta[c]:
                v *= PR['risk']
            L[1] += v
        # tick
        for b2 in range(self.nb):
            j = 4 + 6 * b2
            if L[j + 2]:
                L[j + 2] -= 1
                if L[j + 2] == 0:
                    L[j] = L[j + 1]
                    L[j + 1] = -1
        for s in S.msites:
            if L[JB + s]:
                L[JB + s] -= 1
                if L[JB + s] == 0:
                    L[WH + 3 * s + 2] += 1
        r += 1
        L[0] = r
        if r % 4 == 0:
            for s in S.natural:
                if L[ST + s] < 6:
                    L[ST + s] += 1
        return L

    # ------------------------------------------------------------ evaluation
    def evaluate(self, L):
        S = self.S
        WH, JB = self.WH, self.JB
        r = L[0]
        closed = L[3]
        disc = self.disc
        discm = self.discm
        risk = PR['risk']
        val = L[1]
        cost = S.cost
        n = S.n
        dmin = [99] * n
        for b in range(self.nb):
            i = 4 + 6 * b
            if L[i + 2]:
                e = L[i + 2]
                row = cost[L[i + 1]]
            else:
                e = 0
                row = cost[L[i]]
            for s in range(n):
                v = e + row[s]
                if v < dmin[s]:
                    dmin[s] = v
        # products
        prods = []
        for b in range(self.nb):
            i = 4 + 6 * b
            k = L[i + 5]
            if k:
                loc = L[i] if L[i + 2] == 0 else L[i + 1]
                t = r + L[i + 2]
                for _ in range(k):
                    prods.append((t, loc))
        for s in S.msites:
            k = L[WH + 3 * s + 2]
            for _ in range(k):
                prods.append((r + 1 + dmin[s], s))
            if L[JB + s]:
                prods.append((r + max(L[JB + s], dmin[s]) + 1, s))
        om = [c for c in S.manuf if not (closed >> c) & 1 and r < S.cdl[c]]
        if prods and om:
            prods.sort()
            kp = PR['kp']
            for t, loc in prods:
                best, bv = None, 0.0
                for c in om:
                    arr = t + cost[loc][S.cdest[c]]
                    if arr >= S.cdl[c] or arr >= ROUNDS:
                        continue
                    v = 12.0 * discm[arr]
                    if arr > S.oeta[c]:
                        v *= risk
                    if v > bv:
                        best, bv = c, v
                if best is None:
                    continue
                val += kp * bv
                om.remove(best)
        rm = len(om)
        # raw goods carried
        kr = PR['kr']
        usedT = [0] * self.nb
        usedF = [0] * self.nb
        orw = [c for c in S.raws if not (closed >> c) & 1 and r < S.cdl[c]]
        if orw:
            for b in range(self.nb):
                i = 4 + 6 * b
                loc = L[i] if L[i + 2] == 0 else L[i + 1]
                t = r + L[i + 2]
                best, bv = None, 0.0
                for c in orw:
                    g = S.cgood[c]
                    if L[i + 3 + g] - (usedT[b] if g == T else usedF[b]) < S.cqty[c]:
                        continue
                    arr = t + cost[loc][S.cdest[c]]
                    if arr >= S.cdl[c] or arr >= ROUNDS:
                        continue
                    v = 5.0 * disc[arr] * PR['_rawf']
                    if arr > S.oeta[c]:
                        v *= risk
                    if v > bv:
                        best, bv = c, v
                if best is not None and (rm == 0 or bv > 3.0 * disc[r]):
                    val += kr * bv
                    orw.remove(best)
                    if S.cgood[best] == T:
                        usedT[b] += S.cqty[best]
                    else:
                        usedF[b] += S.cqty[best]
        # materials
        M = self.M
        for s in S.msites:
            if (L[2] >> s) & 1:
                M = s
                break
        built = (L[2] >> M) & 1
        if built:
            val += PR['build']
        if rm > 0 and r < ROUNDS - 6:
            lamp = self.lamp
            mu = PR['mu']
            tl = []
            fl = []
            nu = PR['nu']
            ST = self.ST
            for s in range(n):
                w = WH + 3 * s
                if L[w] or L[w + 1]:
                    f = 1.0 if s == M else mu * lamp[cost[s][M] + dmin[s]]
                    tl.extend([f] * L[w])
                    fl.extend([f] * L[w + 1])
            for s in S.natural:
                k = L[ST + s]
                if k > 0:
                    f = nu * lamp[dmin[s] + 1 + cost[s][M]]
                    if S.roles[s] == T:
                        tl.extend([f] * min(k, 3))
                    else:
                        fl.extend([f] * min(k, 3))
            for b in range(self.nb):
                i = 4 + 6 * b
                cT = L[i + 3] - usedT[b]
                cF = L[i + 4] - usedF[b]
                if cT or cF:
                    if L[i + 2]:
                        d = L[i + 2] + cost[L[i + 1]][M]
                    else:
                        d = cost[L[i]][M]
                    f = lamp[d]
                    if cT > 0:
                        tl.extend([f] * cT)
                    if cF > 0:
                        fl.extend([f] * cF)
            tl.sort(reverse=True)
            fl.sort(reverse=True)
            if not built:
                val += PR['build'] * (sum(tl[:2]) + sum(fl[:1])) / 3.0 - PR['build'] * 0.5
                tl = tl[2:]
                fl = fl[1:]
            te = sum(tl[:2 * rm])
            fe = sum(fl[:rm])
            batches = min(te / 2.0, fe)
            # expected value per batch from the remaining contracts (with professor risk)
            if PR.get('brisk'):
                ev = 0.0
                for c in om:
                    arr = r + 12 + cost[M][S.cdest[c]]
                    v = discm[r + 10]
                    if arr >= S.cdl[c] or arr >= ROUNDS:
                        v *= 0.3
                    elif arr > S.oeta[c]:
                        v *= risk
                    ev += v
                bv12 = 12.0 * ev / rm
            else:
                bv12 = 12.0 * discm[r + 10]
            tleft = ROUNDS - r
            tf = 1.0 if tleft > 25 else max(0.0, (tleft - 6) / 19.0)
            val += tf * (PR['kb'] * bv12 * batches + PR['unit'] * (te + fe) * disc[r])
        return val

    # ------------------------------------------------------------ search
    def search(self, L0, H, W, seed, deadline, hard=None, emax=None):
        beam = [(self.evaluate(L0), L0, None, [])]
        nexp = 0
        for depth in range(H):
            children = {}
            seed_child = None
            for v, L, first, path in beam:
                on_seed = seed is not None and depth < len(seed) and path == seed[:depth]
                for a in self.actions(L):
                    L2 = self.apply(L, a)
                    nexp += 1
                    v2 = self.evaluate(L2)
                    key = tuple(L2[2:])
                    old = children.get(key)
                    if old is None or v2 > old[0]:
                        children[key] = (v2, L2, first if first is not None else a, path + [a])
                    if on_seed and a == seed[depth]:
                        seed_child = key
            if not children:
                break
            items = sorted(children.values(), key=lambda x: -x[0])
            nb = items[:W]
            if seed_child is not None:
                sc = children[seed_child]
                if all(sc[1] is not x[1] for x in nb):
                    nb.append(sc)
            beam = nb
            now = time.perf_counter()
            if emax is not None:
                if nexp > emax and depth >= 2:
                    break
            elif now > deadline and depth >= 2:
                break
            if hard is not None and now > hard:
                break
        best = max(beam, key=lambda x: x[0])
        self.best_value = best[0]
        self.nexp = nexp
        return best[2], best[3]


def est_stock(mem, site, rnd):
    seen = mem['seen'].get(site)
    if seen is None:
        r0, v0 = 0, 4
    else:
        r0, v0 = seen
    regen = rnd // 4 - r0 // 4
    if regen < 0:
        regen = 0
    v = v0 + regen
    return 6 if v > 6 else v


def to_json(a):
    op, b, x, q = a
    if op == WAIT:
        return {'op': 'wait'}
    if op == SAIL:
        return {'op': 'sail', 'boat': b, 'destination': x}
    if op == LOAD:
        return {'op': 'load', 'boat': b, 'good': x, 'quantity': q}
    if op == UNLOAD:
        return {'op': 'unload', 'boat': b, 'good': x, 'quantity': q}
    if op == GATHER:
        return {'op': 'gather', 'boat': b, 'good': x, 'quantity': q}
    if op == BUILD:
        return {'op': 'build_workshop', 'boat': b}
    if op == CRAFT:
        return {'op': 'craft', 'boat': b}
    return {'op': 'deliver', 'boat': b, 'contract': x}


def agent(observation, memory):
    legal = observation['legal_actions']
    try:
        act = decide(observation, memory)
        if act is not None and act in legal:
            return act
        memory['plan'] = None
    except Exception as exc:  # never crash
        print('agent error', repr(exc), file=sys.stderr)
        memory['plan'] = None
    return {'op': 'wait'} if {'op': 'wait'} in legal else legal[0]


def fallback(obs, memory):
    """No search: next step of the stored plan if still legal, else any delivery, else wait."""
    legal = obs['legal_actions']
    plan = memory.get('plan')
    if plan and len(plan) > 1:
        memory['plan'] = plan[1:]
        a = to_json(plan[1])
        if a in legal:
            return a
    memory['plan'] = None
    for a in legal:
        if a['op'] == 'deliver':
            return a
    return {'op': 'wait'}


def decide(obs, memory):
    t0 = time.perf_counter()
    if 'S' not in memory:
        S = Static(obs)
        memory['S'] = S
        memory['seen'] = {}
        memory['M'] = pick_workshop(S)
        memory['plan'] = None
        memory['used'] = 0.0
    S = memory['S']
    rnd = obs['round']
    for s, v in enumerate(obs['stocks']):
        if v is not None and S.roles[s] in (T, F):
            memory['seen'][s] = (rnd, v)
    nb = len(obs['own']['boats'])
    PR['_rawf'] = PR['rawv']
    if rnd >= PR.get('tieb', 99) and obs['own']['score'] <= obs['opponent_score']:
        PR['_rawf'] = 1.0
    if memory['used'] >= PR['hard']:
        memory['guard'] = memory.get('guard', 0) + 1
        return fallback(obs, memory)
    pl = Planner(S, memory['M'], nb, rnd)
    L0 = pl.encode(obs, memory)
    remaining = max(0.05, PR['budget'] - memory['used'])
    hard = t0 + max(0.02, PR['hard'] - memory['used'])
    emax = None
    if PR['ebud']:
        eleft = max(2000, PR['ebud'] - memory.get('exp', 0))
        emax = eleft / max(1, ROUNDS - rnd) * PR.get('dl_f', 1.5)
    per = remaining / max(1, ROUNDS - rnd)
    deadline = t0 + per * PR.get('dl_f', 1.5)
    seed = memory.get('plan')
    if seed:
        seed = seed[1:] if len(seed) > 1 else None
    H, W = PR['H'], PR['W']
    if rnd == 0 and PR.get('H0'):
        H, W = PR['H0'], PR['W0']
        deadline = t0 + PR.get('T0', 1.0)
    if rnd == 0 and PR.get('msel') and len(S.msites) > 1:
        best = None
        for m in S.msites:
            plm = Planner(S, m, nb, rnd)
            f, pth = plm.search(L0, H, W, None, t0 + 2.0, hard, PR['e0'] if PR['ebud'] else None)
            memory['exp'] = memory.get('exp', 0) + plm.nexp
            if best is None or plm.best_value > best[0]:
                best = (plm.best_value, m, f, pth)
        memory['M'] = best[1]
        first, path = best[2], best[3]
    else:
        first, path = pl.search(L0, H, W, seed, deadline, hard, emax)
        memory['exp'] = memory.get('exp', 0) + pl.nexp
    memory['plan'] = path
    memory['used'] += time.perf_counter() - t0
    if first is None:
        return {'op': 'wait'}
    return to_json(first)


def serve():
    protocol = sys.stdout
    sys.stdout = sys.stderr          # print() for debugging goes to stderr, not the protocol
    memory = {}
    for line in sys.stdin:
        observation = json.loads(line)['observation']
        if observation['round'] == 0:
            memory = {}
        protocol.write(json.dumps(agent(observation, memory)) + '\n')
        protocol.flush()


if __name__ == '__main__':
    serve()
