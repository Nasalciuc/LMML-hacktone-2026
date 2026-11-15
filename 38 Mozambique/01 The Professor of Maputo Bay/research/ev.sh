# usage: bash ev.sh agent.py NMAPS SEED TAG [KEY=VAL ...]   -> res_TAG.json, one summary line + race stats
cd ~/moz
A=$1; N=$2; SEED=$3; TAG=$4; shift 4
~/pypy/bin/pypy3 work/evalx2.py $A $N $SEED 12 res_$TAG.json hard=1000 "$@" 2>&1 | head -2
python3 race_stats.py res_$TAG.json | sed -n '2,7p;10,11p'
python3 -c "
import json
r=json.load(open('res_$TAG.json'))
t=sorted(x['think'] for x in r); n=len(t)
print('  think p50 %.2f p99 %.2f max %.2f  bad-action games %d' % (t[n//2], t[int(n*.99)], t[-1], sum(1 for x in r if x['illegal'])))
"
