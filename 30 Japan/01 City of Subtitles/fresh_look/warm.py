import csv, os, numpy as np
from PIL import Image
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
items=[('anonymous/'+f,f,att[f]['photographer'],'anon') for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append(('credited/'+p+'/'+f,f,p,'cred'))
out=[]
for rel,f,who,s in items:
    a=np.asarray(Image.open(D+rel).convert('RGB'),np.float32)
    rowblack=(a.max(2)<3).mean(1)>0.6
    A=a[~rowblack].reshape(-1,3)
    A=A[A.max(1)>4]
    q=np.percentile(A,[0.1,1,50,99,99.9],axis=0)
    lum=A.mean(1)
    hi=A[lum>np.percentile(lum,99.5)]; lo=A[lum<np.percentile(lum,0.5)]
    out.append(dict(file=f,who=who,set=s,**{'p%s_%s'%(p,c):q[i,j] for i,p in enumerate(['01','1','50','99','999']) for j,c in enumerate('RGB')},
        hiRB=(hi[:,0]-hi[:,2]).mean(),hiGB=(hi[:,1]-hi[:,2]).mean(),loRB=(lo[:,0]-lo[:,2]).mean(),loGB=(lo[:,1]-lo[:,2]).mean(),
        hiL=hi.mean(),loL=lo.mean()))
w=csv.DictWriter(open('warm.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.DataFrame(out)
print(d.groupby(['set','who'])[['hiRB','hiGB','loRB','loGB','hiL','loL','p999_R','p999_B','p01_R','p01_B']].agg(['median','std']).round(2).to_string())
t=d[(d.set=='anon')&(d.who=='toorisugari')]
print(t[['file','hiRB','hiGB','loRB','loGB','hiL','loL','p999_R','p999_G','p999_B','p01_R','p01_G','p01_B']].round(1).to_string())
