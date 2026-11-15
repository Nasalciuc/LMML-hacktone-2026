"""OCR of every caption line (manga-ocr, pip install manga-ocr). Finds white-on-black text rows, crops each
line, OCRs it. Writes out/captions_ocr.json {photo: [lines]}. Run with python -X utf8 on Windows."""
from PIL import Image
import numpy as np, os, sys, csv, glob, json
from scipy import ndimage
base=r"C:/Users/user/projects/LMML-hacktone-2026/30 Japan/task_japan/dataset"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
from manga_ocr import MangaOcr
moc=MangaOcr()
def lines_of(path):
    im=Image.open(path).convert('RGB'); a=np.asarray(im).astype(np.int16)
    mn=a.min(2); g=a.mean(2)
    bg=ndimage.uniform_filter(g,size=25)
    text=(mn>175)&(bg<70)
    rowsum=text.sum(1); thr=text.shape[1]*0.02
    on=rowsum>thr; ys=np.where(on)[0]
    if len(ys)==0: return []
    groups=[]; start=ys[0]; prev=ys[0]
    for y in ys[1:]:
        if y-prev<=10: prev=y
        else: groups.append((start,prev)); start=y; prev=y
    groups.append((start,prev))
    bands=[(s,e) for s,e in groups if e-s>=8]
    out=[]
    for (s,e) in bands:
        # x extent of text in this band
        sub=text[s:e+1,:]; xs=np.where(sub.sum(0)>0)[0]
        if len(xs)==0: continue
        x0,x1=xs.min(),xs.max()
        crop=im.crop((max(0,x0-8),max(0,s-6),min(im.width,x1+8),min(im.height,e+6)))
        out.append(moc(crop))
    return out
files=sorted(glob.glob(os.path.join(base,'anonymous','*.jpg')))
files+=sorted(glob.glob(os.path.join(base,'credited','*','*.jpg')))
res={}
for i,p in enumerate(files):
    rel=os.path.relpath(p,base).replace(os.sep,'/')
    res[rel]=lines_of(p)
    if i%20==0: print(i,rel,len(res[rel]),'lines', flush=True)
json.dump(res,open(os.path.join(OUT,'captions_ocr.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=0)
print('DONE',len(res))
