"""c4 = c3 with open=0 by default and a late-game switch: raw contracts get full value from round PR['rawr'] on."""
src = open('c3.py').read()
def rep(a, b):
    global src
    assert a in src, a
    src = src.replace(a, b, 1)
rep("    'open': 1,", "    'open': 0,")
rep("    'rawv': 0.5,", "    'rawv': 0.5,\n    'rawr': 99,         # from this round on raw contracts are worth full value\n")
rep("self.disc[r] * PR['rawv'])", "self.disc[r] * (PR['rawv'] if r < PR['rawr'] else 1.0))")
rep("v = 5.0 * disc[arr] * PR['rawv']", "v = 5.0 * disc[arr] * (PR['rawv'] if arr < PR['rawr'] else 1.0)")
open('c4.py', 'w').write(src)
print('c4 ok')
