src=open('c1.py').read()
def rep(a,b,cnt=1):
    global src
    assert src.count(a)>=1,a
    src=src.replace(a,b,cnt)
rep("    'boat3': -1,", "    'boat3': -1,\n    'rawv': 1.0,        # multiplier on raw contract value\n    'etam': 0,          # 1 = use slot-median professor claim table\n    'etash': 0,         # shift added to the slot table\n")
# professor eta table
rep("        self.oeta = eta\n",
    "        if PR.get('etam'):\n"
    "            tab = [8, 16, 25, 35, 31, 42, 49, 56]\n"
    "            eta = [tab[c] + PR.get('etash', 0) if c < 8 else 99 for c in range(self.nc)]\n"
    "        self.oeta = eta\n")
# raw value multiplier in apply deliver
rep("            v = S.cpts[c] * (self.discm[r] if S.cgood[c] == P else self.disc[r])\n",
    "            v = S.cpts[c] * (self.discm[r] if S.cgood[c] == P else self.disc[r] * PR['rawv'])\n")
rep("                    v = 5.0 * disc[arr]\n", "                    v = 5.0 * disc[arr] * PR['rawv']\n")
open('c2.py','w').write(src)
print('ok')
