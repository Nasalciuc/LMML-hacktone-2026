"""Insert the Professor tables into an agent body. usage: python3 assemble.py body.py tables.txt out.py"""
import sys

body = open(sys.argv[1]).read()
tab = open(sys.argv[2]).read().strip()
head = ("# Professor delivery-round survival (permille) per contract slot and distance; see research/mk_tables.py\n"
        "MLO, MHI, RLO, RHI = 2, 13, 1, 9\n")
assert '#@@TABLES@@' in body
open(sys.argv[3], 'w').write(body.replace('#@@TABLES@@', head + tab))
print('assembled', len(body) + len(tab))
