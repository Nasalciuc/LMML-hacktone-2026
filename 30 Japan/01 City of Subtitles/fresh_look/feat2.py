import csv, os, numpy as np
from PIL import Image
from scipy import ndimage
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
items=[('anonymous/'+f,f,att[f]['photographer'],'anon') for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append(('credited/'+p+'/'+f,f,p,'cred'))
out=[]
for rel,f,who,s in items:
    a=np.asarray(Image.open(D+rel).convert('RGB'),np.float32)
    g=a.mean(2)
    rowblack=(a.max(2)<3).mean(1)>0.6
    # --- caption text colour: bright pixels inside black rows
    band=a[rowblack]
    if band.size:
        bl=band.reshape(-1,3); br=bl[bl.mean(1)>200]
        txt=np.median(br,0) if len(br)>50 else np.array([np.nan]*3)
        txtmax=bl.max(0)
        ntxt=len(br)
    else:
        txt=np.array([np.nan]*3); txtmax=txt; ntxt=0
    # --- grain stats in flat regions of the photo area
    res=a-ndimage.gaussian_filter(a,(2,2,0))
    gm=ndimage.gaussian_gradient_magnitude(ndimage.gaussian_filter(g,1),2)
    content=~rowblack[:,None]&np.ones_like(g,bool)
    thr=np.percentile(gm[content],25)
    m=content&(gm<thr)&(g>25)&(g<230)
    m[:4]=m[-4:]=False; m[:,:4]=m[:,-4:]=False
    rl=res.mean(2)
    def ac(dy,dx):
        x=rl; y=np.roll(rl,(dy,dx),(0,1)); mm=m&np.roll(m,(dy,dx),(0,1))
        return float((x[mm]*y[mm]).mean()/(x[mm]**2).mean())
    cc=np.corrcoef(np.stack([res[...,c][m] for c in range(3)]))
    sd=[float(res[...,c][m].std()) for c in range(3)]
    # grain vs brightness: slope of residual std over luminance bins
    lb=np.clip((g[m]//32).astype(int),0,7); rv=rl[m]
    stds=[rv[lb==k].std() if (lb==k).sum()>500 else np.nan for k in range(8)]
    out.append(dict(file=f,who=who,set=s,txtR=txt[0],txtG=txt[1],txtB=txt[2],txtmaxR=txtmax[0],txtmaxB=txtmax[2],ntxt=ntxt,
        ac01=ac(0,1),ac10=ac(1,0),ac11=ac(1,1),ac02=ac(0,2),ac20=ac(2,0),ccRG=cc[0,1],ccRB=cc[0,2],ccGB=cc[1,2],
        sdR=sd[0],sdG=sd[1],sdB=sd[2],nflat=int(m.sum()),**{'sl%d'%k:stds[k] for k in range(8)}))
w=csv.DictWriter(open('feat2.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('done')
