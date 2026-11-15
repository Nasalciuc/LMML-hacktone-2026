"""final.py = c5 with the measured-best settings, dead experiment paths removed, and a hard time guard.
usage (in ~/moz): python3 mk_final.py c5.py final.py"""
import re
import sys

src = open(sys.argv[1]).read()


def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a)
    src = src.replace(a, b)


def cut(start, end):
    """remove src[start_index : end_index] where start/end are unique markers (end kept)"""
    global src
    i = src.index(start)
    j = src.index(end, i)
    src = src[:i] + src[j:]


# ---- header
rep('''"""Your submission. Upload only this file; keep it self-contained (standard library only).

The evaluator runs `python main.py`, sends one JSON line {"observation": ...} per
decision on stdin and reads one JSON action per line from stdout. Leave the loop
at the bottom as it is and put your strategy in agent().
"""''', '''"""Mozambique - The Professor of Maputo Bay. Submission: this file only (stdlib, PyPy 3.10).

Beam search over our own actions (one action per round for the whole player) on a local
copy of the rules. Raw contracts are valued at half their points inside the search, which
pushes the plan towards manufactured contracts (12 points), the ones that decide the race
against the Professor. A hard time guard keeps a game well inside the 10 s limit.
The serve() loop at the bottom is the template's and must stay as it is.
"""''')

# ---- parameters: rewrite the tail of the PR block (experiment knobs that are off are dropped)
i = src.index("    'budget': 6.0,")
j = src.index("}\n\n# op codes", i)
src = (src[:i] +
       "    'budget': 6.0,      # seconds of search per game (soft, spread over the rounds)\n"
       "    'hard': 8.0,        # seconds per game after which we stop searching (limit is 10)\n"
       "    'rawv': 0.5,        # raw contracts count at this fraction of their points in the search\n"
       "    'tieb': 99,         # from this round on raw contracts count fully while not ahead (99 = off)\n"
       "    '_rawf': 0.5,       # (internal) raw factor in force for the current decision\n"
       + src[j:])

# ---- professor eta table experiment (off)
rep("        if PR.get('etam'):\n"
    "            tab = [8, 16, 25, 35, 31, 42, 49, 56]\n"
    "            eta = [tab[c] + PR.get('etash', 0) if c < 8 else 99 for c in range(self.nc)]\n", "")

# ---- raw factor without the fixed late switch
rep("(PR['_rawf'] if r < PR['rawr'] else 1.0)", "PR['_rawf']")
rep("(PR['_rawf'] if arr < PR['rawr'] else 1.0)", "PR['_rawf']")

# ---- scripted opening experiment (off)
cut("def opening(obs, memory):", "def est_stock(mem, site, rnd):")

# ---- search: hard deadline
rep("    def search(self, L0, H, W, seed, deadline):\n",
    "    def search(self, L0, H, W, seed, deadline, hard=None):\n")
rep("            if time.perf_counter() > deadline and depth >= 2:\n                break\n",
    "            now = time.perf_counter()\n"
    "            if now > deadline and depth >= 2:\n                break\n"
    "            if hard is not None and now > hard:\n                break\n")

# ---- decide: drop boat3/opening branches, add the guard
rep("    if PR.get('boat3', -1) >= 0 and nb < 3 and rnd >= PR['boat3']:\n"
    "        ba = {'op': 'build_boat'}\n"
    "        if ba in obs['legal_actions']:\n"
    "            memory['used'] += time.perf_counter() - t0\n"
    "            return ba\n", "")
rep("    if PR.get('open') and not memory.get('open_done'):\n"
    "        oa = opening(obs, memory)\n"
    "        if oa is None:\n"
    "            memory['open_done'] = True\n"
    "        else:\n"
    "            memory['plan'] = None\n"
    "            memory['used'] += time.perf_counter() - t0\n"
    "            return oa\n",
    "    if memory['used'] >= PR['hard']:\n"
    "        memory['guard'] = memory.get('guard', 0) + 1\n"
    "        return fallback(obs, memory)\n")
rep("    remaining = max(0.05, PR['budget'] - memory['used'])\n",
    "    remaining = max(0.05, PR['budget'] - memory['used'])\n"
    "    hard = t0 + max(0.02, PR['hard'] - memory['used'])\n")
rep("            f, pth = plm.search(L0, H, W, None, t0 + 2.0)\n",
    "            f, pth = plm.search(L0, H, W, None, t0 + 2.0, hard)\n")
rep("        first, path = pl.search(L0, H, W, seed, deadline)\n",
    "        first, path = pl.search(L0, H, W, seed, deadline, hard)\n")

# ---- fallback used once the hard guard trips
rep("def decide(obs, memory):\n", '''def fallback(obs, memory):
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
''')

for bad in ('boat3', 'def opening', 'opening(obs', "PR.get('open')", 'etam', 'rawr', 'etash'):
    assert bad not in src, bad
open(sys.argv[2], 'w').write(src)
print('final ok', len(src), 'bytes')
