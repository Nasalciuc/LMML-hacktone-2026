"""Compose p2_body.py (rollout agent) from the shared parts of p1_body.py and p2_new.txt.
usage (in research/): python mk_p2.py [new.txt] [out_body.py]"""
import sys

newf = sys.argv[1] if len(sys.argv) > 1 else 'p2_new.txt'
outf = sys.argv[2] if len(sys.argv) > 2 else 'p2_body.py'
s = open('p1_body.py').read()
i0 = s.index("    # ------------------------------------------------------------ dynamics\n    def actions(self, L):")
i1 = s.index("    def apply(self, L, a):")
i2 = s.index("    # ------------------------------------------------------------ evaluation")
i3 = s.index("def est_stock(mem, site, rnd):")
i4 = s.index("def agent(observation, memory):")
i5 = s.index("def serve():")
head, apply_part, utils, serve = s[:i0], s[i1:i2], s[i3:i4], s[i5:]
new = open(newf).read()
pr = new[new.index('#@@PR@@\n') + 8:new.index('#@@METHODS@@\n')]
methods = new[new.index('#@@METHODS@@\n') + 13:new.index('#@@AGENT@@\n')]
agent = new[new.index('#@@AGENT@@\n') + 11:]
i = head.index('PR = {')
j = head.index('# op codes')
head = head[:i] + pr + '\n' + head[j:]
head = head.replace("WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)\n",
                    "WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)\n"
                    "# boat goals of the base policy\nG_NONE, G_FETCH, G_TOM, G_DELIV, G_RAW, G_WAITM = range(6)\nNOGOAL = (G_NONE, 0, 0)\n")
doc_end = head.index('"""', 3) + 3
DOC = '''"""Mozambique - The Professor of Maputo Bay. Submission: this file only (stdlib, PyPy 3.10).

Rollout planner. A rule-based base policy gives every boat a goal (fetch materials, bring them
to the workshop, wait there to craft and load, deliver a product, serve a raw contract) and
plays one action per round. Each round we try every alternative goal for the idle boats, play
the rest of the game with the base policy on a local copy of the rules, and keep the best.
A delivery in round t is worth the expected score margin it brings, from survival tables of
the Professor's delivery round (unopposed Professor games, per contract slot and distance
from its home). No wall-clock dependence except a safety net.
"""'''
head = DOC + head[doc_end:]
open(outf, 'w').write(head + apply_part + methods + '\n\n' + utils + agent + '\n\n' + serve)
print('wrote', outf)
