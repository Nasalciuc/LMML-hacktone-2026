import json,collections,statistics as st
G=[json.loads(l) for l in open('wait400.jsonl')]
def claims(g):
    out={}
    for e in g['log']:
        for c,x in enumerate(e['claims']):
            if x!=-1 and c not in out: out[c]=e['r']
    return out
# score by game idx
bygame=collections.defaultdict(list); bytopo=collections.defaultdict(list)
for g in G:
    s=g['result']['scores'][1]; bygame[g['game']].append(s); bytopo[g['group']//2].append(s)
print('prof score by game idx:',{k:round(st.mean(v),1) for k,v in sorted(bygame.items())})
print('prof score by topology:',{k:round(st.mean(v),1) for k,v in sorted(bytopo.items())})
# distinct behaviours per map: how many distinct claim-sequences among 5 games
nd=collections.Counter()
same_as=collections.Counter()
maps=collections.defaultdict(dict)
for g in G: maps[(g['group'],g['seed'])][g['game']]=g
for k,v in maps.items():
    seqs=[tuple(sorted(claims(v[i]).items())) for i in range(5)]
    nd[len(set(seqs))]+=1
    for i in range(5):
        for j in range(i+1,5):
            if seqs[i]==seqs[j]: same_as[(i,j)]+=1
print('distinct claim schedules per map:',dict(nd))
print('identical pairs (i,j):',dict(same_as))
# manuf claims: count and first manuf claim round per game idx
fm=collections.defaultdict(list); nman=collections.defaultdict(list); nraw=collections.defaultdict(list)
for g in G:
    cl=claims(g); cs=g['map']['contracts']
    m=[r for c,r in cl.items() if cs[c]['good']==2]
    rw=[r for c,r in cl.items() if cs[c]['good']!=2]
    nman[g['game']].append(len(m)); nraw[g['game']].append(len(rw))
    if m: fm[g['game']].append(min(m))
print('n manuf claimed by game:',{k:round(st.mean(v),2) for k,v in sorted(nman.items())})
print('n raw claimed by game:',{k:round(st.mean(v),2) for k,v in sorted(nraw.items())})
print('first manuf round by game:',{k:(round(st.mean(v),1),min(v)) for k,v in sorted(fm.items())})
# overall manuf claim rounds, sorted by deadline slot
slot=collections.defaultdict(list)
for g in G:
    cl=claims(g); cs=g['map']['contracts']
    for c in range(8):
        slot[c].append(cl.get(c,99))
for c in range(8):
    v=slot[c]; got=[x for x in v if x<99]
    print('contract slot',c,'claimed %.0f%%'%(100*len(got)/len(v)),'mean r %.1f  p10 %d p50 %d'%(st.mean(got),sorted(got)[len(got)//10],sorted(got)[len(got)//2]))
