import csv, os, numpy as np
from PIL import Image
from scipy import ndimage
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
cap={r['file']:r for r in csv.DictReader(open(R+"01 City of Subtitles/out/caption_linecounts.csv"))}
items=[(f,att[f]['photographer'],'anon') for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append((p+'/'+f,p,'cred'))
K=np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],float)
out=[]
for f,who,s in items:
    path=D+('anonymous/'+f if s=='anon' else 'credited/'+f)
    im=Image.open(path); q=im.quantization
    a=np.asarray(im.convert('RGB'),np.float32)
    g=a.mean(2)
    rowblack=(a.max(2)<3).mean(1)>0.98
    content=~rowblack
    A=a[content]
    # Immerkaer noise per channel in flat areas
    sig=[]
    for c in range(3):
        ch=a[...,c][content]
        x=a[...,c]
        L=ndimage.convolve(x,K)
        grad=np.abs(ndimage.sobel(x,0))+np.abs(ndimage.sobel(x,1))
        m=content[:,None]&(grad<np.percentile(grad[content],30))&(x>20)&(x<235)
        sig.append(np.sqrt(np.pi/2)/6*np.abs(L[m]).mean())
    lum=g[content]
    # chroma vs luma grain: residual high-pass
    hp=a-ndimage.uniform_filter(a,(3,3,1))
    hpl=hp.mean(2); hpc=hp[...,0]-hp[...,2]
    m=content[:,None]&(np.abs(ndimage.sobel(g,0))+np.abs(ndimage.sobel(g,1))<np.percentile((np.abs(ndimage.sobel(g,0))+np.abs(ndimage.sobel(g,1)))[content],30))
    out.append(dict(file=f,who=who,set=s,size=os.path.getsize(path),q0=int(sum(q[0])),q1=int(sum(q.get(1,[0]))),nq=len(q),
      sub=str(im.layer if hasattr(im,'layer') else ''),
      mR=A[:,0].mean(),mG=A[:,1].mean(),mB=A[:,2].mean(),RB=(A[:,0]-A[:,2]).mean(),lum=lum.mean(),lstd=lum.std(),
      minv=A.min(),maxv=A.max(),p1=np.percentile(lum,1),p99=np.percentile(lum,99),
      sR=sig[0],sG=sig[1],sB=sig[2],hpl=hpl[m].std(),hpc=hpc[m].std(),
      blackrows=int(rowblack.sum()),firstblack=int(np.argmax(rowblack)) if rowblack.any() else -1,
      rot=att[f]['rotated180'] if s=='anon' else '',z=att[f]['z'] if s=='anon' else '',
      nlines=cap.get(f,{}).get('nlines',''),paneltop=cap.get(f,{}).get('paneltop',''),textpx=cap.get(f,{}).get('textpx','')))
w=csv.DictWriter(open('feat1.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out))
