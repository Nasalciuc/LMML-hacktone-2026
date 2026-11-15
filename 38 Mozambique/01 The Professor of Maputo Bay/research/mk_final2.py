"""final2.py = final.py with a deterministic search budget counted in node expansions.
PR['ebud'] = expansions per game (0 = old wall-clock mode). Wall clock stays only as a safety net.
usage (in ~/moz): python3 mk_final2.py final.py final2.py"""
import sys

src = open(sys.argv[1]).read()


def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a)
    src = src.replace(a, b)


rep("    'hard': 8.0,", "    'ebud': 0,          # node expansions per game for the search (0 = wall-clock budget)\n    'e0': 60000,        # expansion cap of each round-0 workshop-site search\n    'hard': 8.0,")
# search: count expansions, stop on the expansion cap
rep("    def search(self, L0, H, W, seed, deadline, hard=None):\n        beam = [(self.evaluate(L0), L0, None, [])]\n",
    "    def search(self, L0, H, W, seed, deadline, hard=None, emax=None):\n        beam = [(self.evaluate(L0), L0, None, [])]\n        nexp = 0\n")
rep("                    L2 = self.apply(L, a)\n", "                    L2 = self.apply(L, a)\n                    nexp += 1\n")
rep("            now = time.perf_counter()\n            if now > deadline and depth >= 2:\n                break\n",
    "            now = time.perf_counter()\n"
    "            if emax is not None:\n"
    "                if nexp > emax and depth >= 2:\n                    break\n"
    "            elif now > deadline and depth >= 2:\n                break\n")
rep("        self.best_value = best[0]\n", "        self.best_value = best[0]\n        self.nexp = nexp\n")
# decide: per-decision expansion cap
rep("    hard = t0 + max(0.02, PR['hard'] - memory['used'])\n",
    "    hard = t0 + max(0.02, PR['hard'] - memory['used'])\n"
    "    emax = None\n"
    "    if PR['ebud']:\n"
    "        eleft = max(2000, PR['ebud'] - memory.get('exp', 0))\n"
    "        emax = eleft / max(1, ROUNDS - rnd) * PR.get('dl_f', 1.5)\n")
rep("            f, pth = plm.search(L0, H, W, None, t0 + 2.0, hard)\n",
    "            f, pth = plm.search(L0, H, W, None, t0 + 2.0, hard, PR['e0'] if PR['ebud'] else None)\n"
    "            memory['exp'] = memory.get('exp', 0) + plm.nexp\n")
rep("        first, path = pl.search(L0, H, W, seed, deadline, hard)\n",
    "        first, path = pl.search(L0, H, W, seed, deadline, hard, emax)\n"
    "        memory['exp'] = memory.get('exp', 0) + pl.nexp\n")
open(sys.argv[2], 'w').write(src)
print('final2 ok', len(src), 'bytes')
