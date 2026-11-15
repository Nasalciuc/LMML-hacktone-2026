import os, csv, numpy as np
from PIL import Image
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r['photographer'] for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
items=[('anonymous/'+f,att[f]) for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append(('credited/'+p+'/'+f,p))
V=[]
for rel,who in items:
    im=Image.open(D+rel).convert('L').resize((64,48),Image.BILINEAR)
    a=np.asarray(im,np.float32); V.append(a)
V=np.stack(V)
def norm(x):
    x=x-x.mean((1,2),keepdims=True); return x/np.linalg.norm(x.reshape(len(x),-1),axis=1)[:,None,None]
A=norm(V).reshape(len(V),-1)
best=[]
for name,B in (('id',A),('rot180',norm(V[:,::-1,::-1]).reshape(len(V),-1)),('mirror',norm(V[:,:,::-1]).reshape(len(V),-1))):
    C=A@B.T
    if name=='id': np.fill_diagonal(C,-1)
    iu=np.argwhere(np.triu(C,0 if name!='id' else 1)>0.75)
    for i,j in iu: best.append((C[i,j],name,items[i][0],items[j][0]))
best.sort(reverse=True)
for b in best[:20]: print('%.3f %s %s %s'%b)
print(len(best))
