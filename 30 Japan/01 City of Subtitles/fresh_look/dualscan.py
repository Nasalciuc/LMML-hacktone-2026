import torch, numpy as np, csv
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/"
d=torch.load(R+"cache/feats.pt",mmap=True)
names=d['names']
att={r['file']:r for r in csv.DictReader(open(R+'attribution_table.csv'))}
cre={(r['credited_as'],r['file']):r for r in csv.DictReader(open(R+'credited_check.csv'))}
H6=['amayadori','jihanki','rojiura','shuuden','toorisugari','yakin']
def info(i):
    h,f=names[i].split('/')
    if h=='anonymous': return att[f]['photographer'],int(att[f]['rotated180']),'anon'
    c=cre[(h,f)]; z0=float(c['z_'+h]); z1=float(c['z_'+h+'_rot180'])
    return h,int(z1>z0),'cred'
INFO=[info(i) for i in range(len(names))]
flip=lambda t: torch.flip(t,(-2,-1))
dev='cuda'
def get(i):
    return d['W'][i].float().mean(0).to(dev), d['I'][i].float().mean(0).to(dev)
num={};den={}
for i in range(len(names)):
    h,r,s=INFO[i]; w,I=get(i)
    if r: w,I=flip(w),flip(I)
    k=(h,s); num[k]=num.get(k,0)+w*I; den[k]=den.get(k,0)+I*I
FP={}
for k in num:
    K=num[k]/den[k].clamp(min=1); K=K-K.mean(1,keepdim=True); K=K-K.mean(0,keepdim=True); FP[k]=K.half()
SH=[(97,131),(211,353),(-157,419),(331,-277),(-283,-199),(149,557),(-401,89),(53,-613),(389,241),(-71,-463),(263,29),(-337,509)]
def zs(w,S):
    a=w-w.mean(); a=a/a.norm()
    S=S-S.mean((1,2),keepdim=True); S=S/S.flatten(1).norm(dim=1)[:,None,None]
    raw=(S*a).sum((1,2))
    nul=torch.stack([(torch.roll(S,sh,(1,2))*a).sum((1,2)) for sh in SH])
    return (raw/nul.std(0)).cpu().numpy()
rows=[]
for i in range(len(names)):
    h,r,s=INFO[i]; w,I=get(i)
    keys=[(c,ss) for c in H6 for ss in ('cred','anon')]
    S=torch.stack([FP[k].float()*I for k in keys]+[flip(FP[k].float())*I for k in keys])
    z=zs(w,S)
    row={'name':names[i],'who':h,'rot':r,'set':s}
    for j,k in enumerate(keys):
        row['%s_%s_0'%k]=z[j]; row['%s_%s_1'%k]=z[len(keys)+j]
    rows.append(row)
w=csv.DictWriter(open('dualscan.csv','w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('ok')
