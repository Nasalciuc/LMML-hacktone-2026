import sys, os, csv, numpy as np
sys.path.insert(0,"C:/Users/user/AppData/Local/Temp/claude/ed1cc402544f86e3/f1829eba-eef6-4f45-a143-a9a35335a08f/scratchpad/pylib")
import jpeglib
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r['photographer'] for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
items=[(D+'anonymous/'+f,f,att[f],'anon') for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append((D+'credited/'+p+'/'+f,f,p,'cred'))
res=[]
for path,f,who,s in items:
    im=jpeglib.read_dct(path)
    Y=im.Y  # (bh,bw,8,8)
    if f==items[0][1]: print(Y.shape, im.Cb.shape, im.qt[0][0])
    ac=Y.reshape(-1,64)[:,1:].ravel()
    h={v:int((ac==v).sum()) for v in range(-8,9)}
    # chi-square pair statistic for jsteg (pairs 2k,2k+1 excluding 0,1)
    chi=0; n=0
    for k in range(1,30):
        a=(ac==2*k).sum(); b=(ac==2*k+1).sum()
        a2=(ac==-2*k).sum(); b2=(ac==-2*k-1).sum()
        for x,y in ((a,b),(a2,b2)):
            if x+y>10: e=(x+y)/2; chi+=(x-e)**2/e; n+=1
    nz=ac[(ac!=0)]
    nz1=ac[(ac!=0)&(ac!=1)]
    r=dict(file=f,who=who,set=s,n0=h[0],h1=h[1]+h[-1],h2=h[2]+h[-2],h3=h[3]+h[-3],h4=h[4]+h[-4],
           r21=(h[2]+h[-2])/(h[1]+h[-1]),r32=(h[3]+h[-3])/(h[2]+h[-2]),
           pair23=(h[2]+h[-2])/(h[3]+h[-3]+1),pair45=(h[4]+h[-4])/(h[5]+h[-5]+1),
           asym=(h[1]-h[-1])/(h[1]+h[-1]),chi_per=chi/max(n,1),
           lsb_nz=float((np.abs(nz)&1).mean()),lsb_nz1=float((np.abs(nz1)&1).mean()),
           cb_nz=int((im.Cb.reshape(-1,64)[:,1:]!=0).sum()),cr_nz=int((im.Cr.reshape(-1,64)[:,1:]!=0).sum()))
    res.append(r)
w=csv.DictWriter(open('dct1.csv','w',newline=''),fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
