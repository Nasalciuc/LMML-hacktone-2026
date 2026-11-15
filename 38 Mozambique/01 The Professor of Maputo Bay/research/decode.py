"""Decode evalx result json: per-game manufactured/raw counts for me and opp, worst-of-5 + 3+manuf rate."""
import json,sys
def split(s):
    best=None
    for m in range(5):
        rem=s-12*m
        if rem>=0 and rem%5==0 and rem//5<=4:
            best=(m,rem//5)
    return best or (0,0)
res=json.load(open(sys.argv[1]))
bymap={}
for r in res: bymap.setdefault((r['group'],r['seed']),[]).append(r)
worst=[min(x['hp'] for x in v) for v in bymap.values()]
score=sum(worst)/(2*len(bymap))
my3=0;tot=0;mym=0
for r in res:
    mm,mr=split(r['me']); mym+=mm; tot+=1
    if mm>=3: my3+=1
print('score=%.3f  maps win/draw/loss=%d/%d/%d  my_manuf_avg=%.2f  games_with_3+manuf=%.1f%%'%(
    score,sum(w==2 for w in worst),sum(w==1 for w in worst),sum(w==0 for w in worst),mym/tot,100*my3/tot))
