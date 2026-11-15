import csv, os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
dev='cuda'
def gk(s):
    r=int(3*s); x=torch.arange(-r,r+1,device=dev,dtype=torch.float32); k=torch.exp(-x**2/(2*s*s)); return k/k.sum()
def blur(x,s):
    k=gk(s); n=len(k)
    x=F.conv2d(F.pad(x[None,None],(n//2,n//2,0,0),mode='reflect'),k.view(1,1,1,-1))
    x=F.conv2d(F.pad(x,(0,0,n//2,n//2),mode='reflect'),k.view(1,1,-1,1))
    return x[0,0]
def load(p):
    a=torch.from_numpy(np.asarray(Image.open(p).convert('RGB'),np.float32)).to(dev)
    return a
groups={'T':[], 'O':[], 'Tc':[], 'Oc':[]}
for f,r in att.items():
    groups['T' if r['photographer']=='toorisugari' else 'O'].append((D+'anonymous/'+f,int(r['rotated180'])))
# credited orientation from credited_check.csv
cc=list(csv.DictReader(open(R+"01 City of Subtitles/credited_check.csv")))
for r in cc:
    p=D+'credited/'+r['credited_as']+'/'+r['file']
    # orientation for own hypothesis
    z0=float(r['z_'+r['credited_as']]); z1=float(r['z_'+r['credited_as']+'_rot180'])
    groups['Tc' if r['credited_as']=='toorisugari' else 'Oc'].append((p,int(z1>z0)))
res={}
for gname,lst in groups.items():
    for frame in ('posted','sensor'):
        for s in (8,32):
            S=torch.zeros(960,1280,3,device=dev); C=torch.zeros(960,1280,1,device=dev)
            for p,rot in lst:
                a=load(p)
                g=a.mean(2)
                valid=(a.max(2).values>8).float()
                # erode valid by blur so band edges/captions are excluded
                valid=(blur(valid,6)>0.999).float()
                # text exclusion: very bright thin strokes inside dark boxes are rare; skip
                out=[]
                for c in range(3):
                    x=a[...,c]
                    num=blur(x*valid,s); den=blur(valid,s).clamp(min=1e-3)
                    hp=(x-num/den)*valid
                    out.append(hp)
                hp=torch.stack(out,-1)
                v=valid[...,None]
                if frame=='sensor' and rot:
                    hp=torch.flip(hp,(0,1)); v=torch.flip(v,(0,1))
                S+=hp; C+=v
            M=(S/C.clamp(min=1)).cpu().numpy(); res[(gname,frame,s)]=(M,C.cpu().numpy())
            np.save('lm_%s_%s_%d.npy'%(gname,frame,s),M)
            print(gname,frame,s,len(lst),float(np.abs(M).mean()),flush=True)
