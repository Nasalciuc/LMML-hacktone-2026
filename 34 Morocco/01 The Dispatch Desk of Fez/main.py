"""Router for the Fez dispatch desk.

Reads test-input.csv, writes routing_log.csv (request_id,model).
Standard library only.

Rules come from the recorded answers on the 400 training requests:
- price lookups on a board are easy for the apprentice, unless the price is
  reached through another stall ("sells whatever ...");
- shortest walk: when the direct path is already the shortest, the master
  gets it; when a detour is shorter both guides mostly miss it, so we don't
  pay for the master there;
- small multiplications (x10, or few crates and a small product) the
  apprentice gets right;
- takings after a per-crate fee: the master is right only about half the
  time, the apprentice a quarter; not worth 25x the price within the budget;
- everything else (tolls, closing times, plain takings) goes to the master.
"""
import csv
import heapq
import re
import sys

NAME = r"[A-Z][a-z]+"


def direct_is_shortest(prompt):
    edges = re.findall(r"(%s)-(%s) (\d+) minutes" % (NAME, NAME), prompt)
    m = re.search(r"from (%s) to (%s)" % (NAME, NAME), prompt)
    if not edges or not m:
        return None
    src, dst = m.groups()
    graph = {}
    for a, b, w in edges:
        w = int(w)
        graph.setdefault(a, {})[b] = min(w, graph.get(a, {}).get(b, w))
        graph.setdefault(b, {})[a] = min(w, graph.get(b, {}).get(a, w))
    direct = graph.get(src, {}).get(dst)
    if direct is None:
        return False
    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, d):
            continue
        for v, w in graph.get(u, {}).items():
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return direct <= dist.get(dst, float("inf"))


def route(prompt):
    p = prompt
    if "board lists" in p:
        return "expensive" if "sells whatever" in p else "cheap"
    if "Walking times" in p or "shortest walk" in p:
        return "expensive" if direct_is_shortest(p) else "cheap"
    if " loads " in p and "crates of" in p and "sells" not in p and "hands" not in p:
        m = re.search(r"(\d+) crates of .*? with (\d+) ", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 10 in (a, b) or (a <= 13 and a * b <= 100):
                return "cheap"
    if "stall fee" in p:
        return "cheap"
    return "expensive"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "test-input.csv"
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with open("routing_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["request_id", "model"])
        for r in rows:
            try:
                model = route(r.get("prompt") or "")
            except Exception:
                model = "expensive"
            w.writerow([r["request_id"], model])


if __name__ == "__main__":
    main()
