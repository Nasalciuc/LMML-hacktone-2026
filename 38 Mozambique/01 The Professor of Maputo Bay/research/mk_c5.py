"""c5 = c4 + score-aware tie-breaker: from round PR['tieb'] on, if we are not ahead on the observed
scores, raw contracts count at full value again (any extra contract can break a tie)."""
src = open('c4.py').read()
def rep(a, b, n=1):
    global src
    assert src.count(a) >= n, a
    src = src.replace(a, b) if n == 'all' else src.replace(a, b, n)
rep("    'rawr': 99,", "    'rawr': 99,\n    'tieb': 99,         # from this round on, raw contracts count fully while we are not ahead\n    '_rawf': 0.5,       # (internal) raw factor in force for this decision\n")
src = src.replace("(PR['rawv'] if r < PR['rawr'] else 1.0)", "(PR['_rawf'] if r < PR['rawr'] else 1.0)")
src = src.replace("(PR['rawv'] if arr < PR['rawr'] else 1.0)", "(PR['_rawf'] if arr < PR['rawr'] else 1.0)")
assert src.count("PR['_rawf']") == 2
rep("    nb = len(obs['own']['boats'])\n",
    "    nb = len(obs['own']['boats'])\n"
    "    PR['_rawf'] = PR['rawv']\n"
    "    if rnd >= PR.get('tieb', 99) and obs['own']['score'] <= obs['opponent_score']:\n"
    "        PR['_rawf'] = 1.0\n")
open('c5.py', 'w').write(src)
print('c5 ok')
