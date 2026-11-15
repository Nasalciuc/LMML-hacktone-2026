import torch, numpy as np, csv, torch.nn.functional as F
from PIL import Image, ImageDraw
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/"
d=torch.load(R+"cache/feats.pt",mmap=True)
names=d['names']
att={r['file']:r for r in csv.DictReader(open(R+'attribution_table.csv'))}
cre={(r['credited_as'],r['file']):r for r in csv.DictReader(open(R+'credited_check.csv'))}
def info(i):
    h,f=names[i].split('/')
    if h=='anonymous': return att[f]['photographer'],int(att[f]['rotated180']),'anon'
    c=cre[(h,f)]; z0=float(c['z_'+h]); z1=float(c['z_'+h+'_rot180'])
    return h,int(z1>z0),'cred'
INFO=[info(i) for i in range(len(names))]
flip=lambda t: torch.flip(t,(-2,-1))
dev='cuda'
def get(i): return d['W'][i].float().mean(0).to(dev), d['I'][i].float().mean(0).to(dev), d['M'][i].float().to(dev)
H6=['amayadori','jihanki','rojiura','shuuden','toorisugari','yakin']
num={h:0 for h in H6}; den={h:0 for h in H6}
for i in range(len(names)):
    h,r,s=INFO[i]; w,I,m=get(i)
    if r: w,I=flip(w),flip(I)
    num[h]=num[h]+w*I; den[h]=den[h]+I*I
def box(x,k): return F.avg_pool2d(x[None,None],k,k)[0,0]
res={}
for h in H6:
    for s in ('anon','cred'):
        for frame in ('posted','sensor'):
            P=0; Q=0
            for i in range(len(names)):
                hh,r,ss=INFO[i]
                if hh!=h or ss!=s: continue
                w,I,m=get(i)
                ws,Is=(flip(w),flip(I)) if r else (w,I)
                K=(num[h]-ws*Is)/(den[h]-Is*Is).clamp(min=1)
                K=K-K.mean(1,keepdim=True); K=K-K.mean(0,keepdim=True)
                k=flip(K) if r else K
                pred=k*I
                # local normalised product: w*pred / global scale
                prod=w*pred/(pred.std()*w.std())
                mm=m
                if frame=='sensor' and r: prod,mm=flip(prod),flip(mm)
                P=P+prod*mm; Q=Q+mm
            res[(h,s,frame)]=(P.cpu(),Q.cpu())
torch.save(res,'presence_pool.pt')
# render: block means at 16 and 32 px, z-scored by block count
tiles=[]
for (h,s,frame),(P,Q) in res.items():
    if s!='anon' and h!='toorisugari': continue
    for b in (16,32):
        Pb=box(P,b); Qb=box(Q,b)
        z=(Pb/Qb.clamp(min=1e-3))*torch.sqrt(Qb*b*b)  # mean product * sqrt(n): approx z if per-pixel products unit var
        z=z.numpy(); z=(z-np.median(z))/ (np.percentile(z,84)-np.percentile(z,16)+1e-9)*2
        x=np.clip((z+3)/6,0,1)
        im=Image.fromarray((x*255).astype(np.uint8)).resize((480,360),Image.NEAREST)
        ImageDraw.Draw(im).text((3,3),'%s %s %s b%d'%(h[:5],s,frame,b),fill=255)
        tiles.append(im)
cols=4
sheet=Image.new('L',(cols*480,((len(tiles)+cols-1)//cols)*360))
for i,t in enumerate(tiles): sheet.paste(t,((i%cols)*480,(i//cols)*360))
sheet.save('presence_pool.png'); print(len(tiles))
