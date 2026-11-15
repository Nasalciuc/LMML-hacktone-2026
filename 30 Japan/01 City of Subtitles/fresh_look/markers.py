import os, hashlib, csv, collections
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r['photographer'] for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
paths=[(D+'anonymous/'+f,att[f]) for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): paths.append((D+'credited/'+p+'/'+f,p+'*'))
sig=collections.Counter(); per=collections.defaultdict(collections.Counter)
for path,who in paths:
    b=open(path,'rb').read(); i=2; segs=[]; dht=b''
    while i<len(b):
        assert b[i]==0xFF
        m=b[i+1]
        if m==0xDA:
            L=int.from_bytes(b[i+2:i+4],'big'); segs.append('DA'); break
        L=int.from_bytes(b[i+2:i+4],'big')
        seg=b[i+4:i+2+L]
        name='%02X'%m
        if m==0xC4: dht+=seg
        if m in (0xC0,0xC1,0xC2): name+=':'+seg[5:].hex()
        if m==0xE0: name+=':'+seg.hex()
        segs.append(name); i+=2+L
    key=' '.join(segs)+' DHT'+hashlib.md5(dht).hexdigest()[:6]
    sig[key]+=1; per[who][key]+=1
for k,v in sig.items(): print(v,k)
for w in per: print(w, dict((k[-9:],v) for k,v in per[w].items()))
