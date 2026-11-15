"""Mozambique - The Professor of Maputo Bay. Submission: this file only (stdlib, PyPy 3.10).

Beam search over our own actions (one action per round for the whole player) on a local
copy of the rules. Every contract is valued by the expected score margin of delivering it
at a given round, using survival tables of the Professor's delivery round (measured on
10000 unopposed Professor games, per contract slot and distance from its home). States are
ranked by delivered value plus a time-based estimate of the manufacturing pipeline.
The search budget is counted in node expansions, so the agent plays the same game on any
machine; the wall clock is only a safety net. The serve() loop is the template's.
"""
import json
import sys
import time

T, F, P = 0, 1, 2
CAP = 3
ROUNDS = 80
INF = 10 ** 6

PR = {
    'rho': 1.0,         # weight of denying the Professor a contract it would otherwise take
    'rawv': 1.0,        # multiplier on raw contract value
    'gam': 0.995,       # mild per-round discount (breaks ties towards acting sooner)
    'k_boat': 0.97,     # confidence factors of products by stage: on a boat,
    'k_prod': 0.92,     # in the workshop warehouse or in production,
    'k_mat': 0.82,      # still materials that exist (at the workshop or on a boat),
    'k_pro': 0.62,      # materials that still have to be fetched
    'kr': 0.9,          # carried raw goods matched to a raw contract
    'kraw': 0.0,        # raw contracts that could be served from home stock / natural sites
    'rgate': 2.0,       # minimum value for a carried pair of goods to count for a raw contract
    'trip': 4,          # player actions assumed per future fetch trip
    'ntrip': 3,         # future fetch trips considered in the estimate
    'shift': 0,         # rounds subtracted from the Professor's delivery rounds (safety margin)
    'H': 14,            # search horizon
    'W': 24,            # beam width
    'ebud': 90000,      # node expansions per game
    'e0': 4000,         # expansions of each round-0 workshop-site search
    'dl_f': 1.5,        # a decision may use this multiple of the even share of the budget
    'hard': 8.5,        # seconds per game after which we stop searching (limit is 10)
}

# op codes
WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)

#@@TABLES@@


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
                nxt[s][v] = first[v]
        self.dist, self.nxt = dist, nxt
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
        self.surv = [self.professor(c) for c in range(self.nc)]

    def professor(self, c):
        """P(the Professor has not delivered contract c before round t), t = 0..ROUNDS."""
        oh = self.ohome
        d = self.dist
        if self.cgood[c] == P:
            slots = [x for x in self.manuf]
            k = 4 + min(3, slots.index(c))
            f = min([d[oh][m] + d[m][self.cdest[c]] for m in self.msites] or [MHI])
            f = max(MLO, min(MHI, f)) - MLO
        else:
            k = min(3, self.raws.index(c))
            f = max(RLO, min(RHI, d[oh][self.cdest[c]])) - RLO
        row = SURV[k][f] if k < len(SURV) and f < len(SURV[k]) else [1000] * 81
        sh = PR['shift']
        out = []
        for t in range(ROUNDS + 2):
            tt = min(80, max(0, t + sh))
            out.append(row[tt] / 1000.0)
        return out


def pick_workshop(S):
    best, bs = None, None
    for s in S.msites:
        sc = 2.0 * S.dist[S.home][s]
        for c in S.manuf:
            sc += 0.4 * S.dist[s][S.cdest[c]]
        nt = min([S.dist[s][x] for x in S.tsites] or [9])
        nf = min([S.dist[s][x] for x in S.fsites] or [9])
        sc += 1.0 * nt + 0.6 * nf
        if bs is None or sc < bs:
            best, bs = s, sc
    return best


class Planner(object):
    def __init__(self, S, M, nb, r0, claims=None):
        self.S = S
        self.M = M
        self.nb = nb
        self.r0 = r0
        n = S.n
        self.WH = 4 + 6 * nb
        self.JB = self.WH + 3 * n
        self.ST = self.JB + n
        self.size = self.ST + n
        # value of delivering contract c in round t, seen from round r0
        rho = PR['rho']
        gam = PR['gam']
        self.Wt = Wt = []
        for c in range(S.nc):
            sv = S.surv[c]
            s0 = max(0.05, sv[min(r0, ROUNDS)])
            sinf = min(sv[ROUNDS], s0) / s0
            pts = S.cpts[c] * (PR['rawv'] if S.cgood[c] != P else 1.0)
            dl = S.cdl[c]
            row = []
            for t in range(ROUNDS + 1):
                if t >= dl or t >= ROUNDS or t < r0:
                    row.append(0.0)
                    continue
                st = min(sv[t], s0) / s0
                w = pts * ((1.0 + rho) * st - rho * min(sinf, st)) * gam ** (t - r0)
                row.append(max(w, 0.02 * pts))
            Wt.append(row)

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
        M = self.M
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
            for c in S.dest_contracts[p]:
                if not (closed >> c) & 1 and r < S.cdl[c] and L[i + 3 + S.cgood[c]] >= S.cqty[c]:
                    acts.append((DELIVER, b, c, 0))
            role = S.roles[p]
            w = WH + 3 * p
            if p == M:
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
            elif role != 2:
                if free:
                    for g in (T, F):
                        x = L[w + g]
                        if x:
                            for q in range(1, min(x, free) + 1):
                                acts.append((LOAD, b, g, q))
                if (role == T or role == F) and free:
                    x = L[ST + p]
                    if x > 0:
                        q = min(x, free)
                        acts.append((GATHER, b, role, q))
                        if q > 2:
                            acts.append((GATHER, b, role, 2))
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
            L[1] += self.Wt[c][r]
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
        nb = self.nb
        WH, JB, ST = self.WH, self.JB, self.ST
        M = self.M
        r = L[0]
        closed = L[3]
        dist = S.dist
        dM = dist[M]
        Wt = self.Wt
        cdest = S.cdest
        val = L[1]
        if r >= ROUNDS:
            return val
        # boats: round they are free and where
        bt = [0] * nb
        bl = [0] * nb
        atM = INF
        for b in range(nb):
            i = 4 + 6 * b
            if L[i + 2]:
                t = r + L[i + 2]
                p = L[i + 1]
            else:
                t = r
                p = L[i]
            bt[b] = t
            bl[b] = p
            if t + dM[p] < atM:
                atM = t + dM[p]
        cdl = S.cdl
        om = [c for c in S.manuf if not (closed >> c) & 1 and r < cdl[c]]
        orw = [c for c in S.raws if not (closed >> c) & 1 and r < cdl[c]]
        # ---- carried raw goods matched to raw contracts
        usedT = [0] * nb
        usedF = [0] * nb
        if orw:
            kr = PR['kr']
            gate = PR['rgate'] if om else 0.0
            for b in range(nb):
                i = 4 + 6 * b
                if not (L[i + 3] or L[i + 4]):
                    continue
                best, bv = None, gate
                for c in orw:
                    g = S.cgood[c]
                    if L[i + 3 + g] < S.cqty[c]:
                        continue
                    arr = bt[b] + dist[bl[b]][cdest[c]]
                    if arr >= ROUNDS:
                        continue
                    v = Wt[c][arr]
                    if v > bv:
                        best, bv = c, v
                if best is not None:
                    val += kr * bv
                    orw.remove(best)
                    if S.cgood[best] == T:
                        usedT[b] = S.cqty[best]
                    else:
                        usedF[b] = S.cqty[best]
        # ---- raw contracts that could still be served (home stock or a natural site)
        if orw and PR['kraw']:
            kraw = PR['kraw']
            h = S.home
            for c in orw:
                g = S.cgood[c]
                dst = cdest[c]
                arr = INF
                if L[WH + 3 * h + g] >= S.cqty[c]:
                    for b in range(nb):
                        t = bt[b] + dist[bl[b]][h] + 1 + dist[h][dst]
                        if t < arr:
                            arr = t
                for s in (S.tsites if g == T else S.fsites):
                    if L[ST + s] >= S.cqty[c]:
                        for b in range(nb):
                            t = bt[b] + dist[bl[b]][s] + 1 + dist[s][dst]
                            if t < arr:
                                arr = t
                if arr < ROUNDS:
                    val += kraw * Wt[c][arr]
        if not om:
            return val
        # ---- products: (round available for delivery, site, confidence)
        prods = []
        nprod = 0
        k_boat = PR['k_boat']
        for b in range(nb):
            k = L[4 + 6 * b + 5]
            for _ in range(k):
                prods.append((bt[b], bl[b], k_boat))
            nprod += k
        k_prod = PR['k_prod']
        w = WH + 3 * M
        pick = (atM if atM > r else r) + 1
        for _ in range(L[w + 2]):
            prods.append((pick, M, k_prod))
        nprod += L[w + 2]
        job = L[JB + M]
        if job:
            t = r + job
            prods.append(((t if t > atM else atM) + 1, M, k_prod))
            nprod += 1
        need = len(om) - nprod
        if need > 0:
            built = (L[2] >> M) & 1
            nbuild = 0 if built else 1
            reqT = 2 * (need + nbuild)
            reqF = need + nbuild
            tl = [r] * L[w]
            fl = [r] * L[w + 1]
            vt = bt[:]
            vl = bl[:]
            for b in range(nb):
                i = 4 + 6 * b
                cT = L[i + 3] - usedT[b]
                cF = L[i + 4] - usedF[b]
                if cT > 0 or cF > 0:
                    a = bt[b] + dM[bl[b]] + 1
                    if cT > 0:
                        tl.extend([a] * cT)
                    if cF > 0:
                        fl.extend([a] * cF)
                    vt[b] = a
                    vl[b] = M
            nreal_t = len(tl)
            nreal_f = len(fl)
            # future fetch trips, greedy: the good that is short, the boat/source that brings it first
            if nreal_t < reqT or nreal_f < reqF:
                h = S.home
                src = []
                if L[WH + 3 * h] > 0 and h != M:
                    src.append([h, T, L[WH + 3 * h]])
                if L[WH + 3 * h + 1] > 0 and h != M:
                    src.append([h, F, L[WH + 3 * h + 1]])
                for s in S.natural:
                    x = L[ST + s]
                    if x > 0:
                        src.append([s, S.roles[s], x])
                slot = r
                trip = PR['trip']
                for _ in range(PR['ntrip']):
                    dT = reqT - len(tl)
                    dF = reqF - len(fl)
                    if dT <= 0 and dF <= 0:
                        break
                    g = T if dT > 0 and (dF <= 0 or dT >= 2 * dF) else F
                    best, ba, bb = None, INF, 0
                    for e in src:
                        if e[1] != g or e[2] <= 0:
                            continue
                        s = e[0]
                        ds = dM[s] + 2
                        for b in range(nb):
                            a = vt[b] + dist[vl[b]][s] + ds
                            if a < ba:
                                best, ba, bb = e, a, b
                    if best is None:
                        break
                    slot += trip
                    if ba < slot:
                        ba = slot
                    q = best[2] if best[2] < 3 else 3
                    best[2] -= q
                    if g == T:
                        tl.extend([ba] * q)
                    else:
                        fl.extend([ba] * q)
                    vt[bb] = ba
                    vl[bb] = M
            # build, then one batch after another
            k_mat = PR['k_mat']
            k_pro = PR['k_pro']
            it = 0
            jf = 0
            ok = True
            nt = len(tl)
            nf = len(fl)
            if built:
                tfree = r + job
            else:
                if nt >= 2 and nf >= 1:
                    t = tl[1] if tl[1] > fl[0] else fl[0]
                    if atM > t:
                        t = atM
                    tfree = t + 1
                    it = 2
                    jf = 1
                else:
                    ok = False
            if ok:
                while need > 0 and it + 2 <= nt and jf < nf:
                    t = tl[it + 1] if tl[it + 1] > fl[jf] else fl[jf]
                    if tfree > t:
                        t = tfree
                    if atM > t:
                        t = atM
                    conf = k_mat if (it + 2 <= nreal_t and jf < nreal_f) else k_pro
                    tfree = t + 3          # craft in round t, product in the warehouse at t + 3
                    prods.append((tfree + 1, M, conf))
                    it += 2
                    jf += 1
                    need -= 1
        # ---- assign products to open manufactured contracts, earliest product first
        if prods:
            if len(prods) > 1:
                prods.sort()
            for t, loc, conf in prods:
                best, bv = None, 0.0
                row = dist[loc]
                for c in om:
                    arr = t + row[cdest[c]]
                    if arr >= ROUNDS:
                        continue
                    v = Wt[c][arr]
                    if v > bv:
                        best, bv = c, v
                if best is None:
                    continue
                val += conf * bv
                om.remove(best)
                if not om:
                    break
        return val

    # ------------------------------------------------------------ search
    def search(self, L0, H, W, seed, emax, hard):
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
            if nexp > emax and depth >= 2:
                break
            if time.perf_counter() > hard:
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
        memory['exp'] = 0
    S = memory['S']
    rnd = obs['round']
    for s, v in enumerate(obs['stocks']):
        if v is not None and S.roles[s] in (T, F):
            memory['seen'][s] = (rnd, v)
    if memory['used'] >= PR['hard']:
        memory['guard'] = memory.get('guard', 0) + 1
        return fallback(obs, memory)
    hard = t0 + max(0.02, PR['hard'] - memory['used'])
    nb = len(obs['own']['boats'])
    for s, x in enumerate(obs['own']['workshops']):
        if x:
            memory['M'] = s
            memory['msel'] = True
    eleft = max(2000, PR['ebud'] - memory['exp'])
    emax = eleft / max(1, ROUNDS - rnd) * PR['dl_f']
    seed = memory.get('plan')
    if seed:
        seed = seed[1:] if len(seed) > 1 else None
    H, W = PR['H'], PR['W']
    if not memory.get('msel') and len(S.msites) > 1:
        best = None
        for m in S.msites:
            plm = Planner(S, m, nb, rnd)
            L0 = plm.encode(obs, memory)
            f, pth = plm.search(L0, H, W, None, PR['e0'], hard)
            memory['exp'] += plm.nexp
            if best is None or plm.best_value > best[0]:
                best = (plm.best_value, m, f, pth)
        memory['M'] = best[1]
        memory['msel'] = True
        first, path = best[2], best[3]
    else:
        pl = Planner(S, memory['M'], nb, rnd)
        L0 = pl.encode(obs, memory)
        first, path = pl.search(L0, H, W, seed, emax, hard)
        memory['exp'] += pl.nexp
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
