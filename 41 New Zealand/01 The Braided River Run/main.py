"""The Braided River Run: exact shortest-time route by A* search.

State = (cell, bridges down) on ordinary cells, plus (time mod tide period)
on fords. On an ordinary cell you may always wait, so arriving earlier is
never worse and the tide phase can be dropped there; the waits needed to hit
low tide on the next ford are generated as part of the move. The heuristic is
the walking time to G with every bridge down and every ford dry, which never
overestimates, so the first time G comes off the queue the route is optimal.

Usage:  python main.py [map file]      (default test-input.txt -> solution.txt)
"""

import heapq
import os
import sys

import environment as env


def solve(valley):
    H, W = valley.height, valley.width
    P, K = valley.tide_period, valley.tide_low
    grid = valley.grid
    N = H * W

    bridges = sorted(valley.bridges)
    bit = {b: 1 << i for i, b in enumerate(bridges)}
    start_mask = 0
    for b in valley.initial_open:
        start_mask |= bit[b]

    # per-cell info
    kind = [0] * N      # 0 blocked, 1 ground/swamp/hut/winch, 2 ford, 3 bridge
    need = [0] * N      # bridge bit required
    flip = [0] * N      # winch mask
    for r in range(H):
        for c in range(W):
            ch = grid[r][c]
            i = r * W + c
            if ch in ("#", "="):
                kind[i] = 0
            elif ch == "~":
                kind[i] = 2
            elif env.is_bridge(ch):
                kind[i] = 3
                need[i] = bit[ch]
            else:
                kind[i] = 1
                if ch in valley.winches:
                    m = 0
                    for b in valley.winches[ch]:
                        m ^= bit[b]
                    flip[i] = m

    moves = (("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1))
    nbrs = [[] for _ in range(N)]
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if not kind[i]:
                continue
            for a, dr, dc in moves:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < H and 0 <= c2 < W and kind[r2 * W + c2]:
                    nbrs[i].append((a, r2 * W + c2, valley.duration(r, c, r2, c2)))

    # heuristic: reverse Dijkstra from G, everything passable
    goal = valley.goal[0] * W + valley.goal[1]
    INF = float("inf")
    h = [INF] * N
    h[goal] = 0
    pq = [(0, goal)]
    rev = [[] for _ in range(N)]
    for u in range(N):
        for _a, v, d in nbrs[u]:
            rev[v].append((u, d))
    while pq:
        d0, v = heapq.heappop(pq)
        if d0 > h[v]:
            continue
        for u, d in rev[v]:
            if d0 + d < h[u]:
                h[u] = d0 + d
                heapq.heappush(pq, (d0 + d, u))

    def key(cell, mask, t):
        return (cell, mask, t % P) if kind[cell] == 2 else (cell, mask)

    s = valley.start[0] * W + valley.start[1]
    k0 = key(s, start_mask, 0)
    best = {k0: 0}
    parent = {k0: None}
    pq = [(h[s], 0, s, start_mask, k0)]
    while pq:
        f, t, u, mask, ku = heapq.heappop(pq)
        if best.get(ku, INF) < t:
            continue
        if u == goal:
            path = []
            k = ku
            while parent[k] is not None:
                k, acts = parent[k]
                path.append(acts)
            return t, "".join(reversed(path))

        def push(v, nmask, nt, acts):
            kv = key(v, nmask, nt)
            if nt < best.get(kv, INF):
                best[kv] = nt
                parent[kv] = (ku, acts)
                heapq.heappush(pq, (nt + h[v], nt, v, nmask, kv))

        on_ford = kind[u] == 2
        if on_ford and (t + 1) % P < K:
            push(u, mask, t + 1, "W")
        for a, v, d in nbrs[u]:
            kv = kind[v]
            if kv == 3 and not (mask & need[v]):
                continue
            nmask = mask ^ flip[v]
            if kv == 2:
                waits = (0,) if on_ford else range(P)
                for w in waits:
                    arr = t + w + d
                    if arr % P < K:
                        push(v, nmask, arr, "W" * w + a)
            else:
                push(v, nmask, t + d, a)
    return None, None


def main():
    map_path = sys.argv[1] if len(sys.argv) > 1 else "test-input.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "solution.txt"
    valley = env.load_map(map_path)
    minutes, route = solve(valley)
    if route is None:
        print("no route found", file=sys.stderr)
        return 1
    check = env.simulate(valley, route)[-1][0]
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# {} minutes, {} actions\n{}\n".format(check, len(route), route))
    print("route: {} minutes ({} actions)".format(check, len(route)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
