import csv, numpy as np, sys
from PIL import Image, ImageDraw
from scipy import ndimage
D="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/task_japan/dataset/anonymous/"
T="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/attribution_table.csv"
who=sys.argv[1]; mode=sys.argv[2]
rows=[r for r in csv.DictReader(open(T)) if r['photographer']==who]
K=np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],float)
tw,th=320,240
sheet=Image.new('L',(9*tw,int(np.ceil(len(rows)/9))*th))
for i,r in enumerate(rows):
    a=np.asarray(Image.open(D+r['file']).convert('RGB'),np.float32)
    if mode=='luma': x=a.mean(2)
    elif mode=='chroma': x=a[...,0]-a[...,2]
    L=np.abs(ndimage.convolve(x,K))
    m=ndimage.median_filter(L,size=9)          # local grain amplitude (robust)
    g=np.log(m+0.5)
    lo,hi=np.percentile(g,[2,98]); g=np.clip((g-lo)/(hi-lo+1e-6),0,1)
    im=Image.fromarray((g*255).astype(np.uint8)).resize((tw,th),Image.BILINEAR)
    ImageDraw.Draw(im).text((3,2),r['file'][:7],fill=255)
    sheet.paste(im,((i%9)*tw,(i//9)*th))
sheet.save('grain_%s_%s.png'%(who,mode))
