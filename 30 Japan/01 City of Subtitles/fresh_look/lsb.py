import sys, os, csv, re, numpy as np
sys.path.insert(0,"C:/Users/user/AppData/Local/Temp/claude/ed1cc402544f86e3/f1829eba-eef6-4f45-a143-a9a35335a08f/scratchpad/pylib")
import jpeglib
R="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/"
D=R+"task_japan/dataset/"
att={r['file']:r['photographer'] for r in csv.DictReader(open(R+"01 City of Subtitles/attribution_table.csv"))}
items=[(D+'anonymous/'+f,f,att[f]) for f in sorted(att)]
for p in sorted(os.listdir(D+'credited')):
    for f in sorted(os.listdir(D+'credited/'+p)): items.append((D+'credited/'+p+'/'+f,p+'/'+f,p))
ZZ=np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])
def streams(im):
    out={}
    for cn,C in (('Y',im.Y),('Cb',im.Cb),('Cr',im.Cr)):
        nat=C.reshape(-1,64)          # jpeglib gives [.., 8, 8] natural order
        zz=nat[:,ZZ]
        for oname,arr in (('zz',zz),('nat',nat)):
            a=arr[:,1:].ravel()
            out[(cn,oname,'jsteg')]=(np.abs(a[(a!=0)&(a!=1)]) & 1)
            out[(cn,oname,'nz')]=(np.abs(a[a!=0]) & 1)
            out[(cn,oname,'raw')]=(a&1)
        out[(cn,'dc','all')]=(nat[:,0]&1)
    return out
pat=re.compile(rb'[ -~]{10,}|SIGMOID|sigmoid|FLAG|flag')
hits=[]
for path,f,who in items:
    im=jpeglib.read_dct(path)
    for k,b in streams(im).items():
        b=b.astype(np.uint8)
        for inv in (0,1):
            bb=b^inv
            for msb in (1,0):
                for off in range(8):
                    x=bb[off:off+8*4000]; x=x[:len(x)//8*8]
                    by=np.packbits(x,bitorder='big' if msb else 'little').tobytes()
                    for m in pat.finditer(by[:4000]):
                        hits.append((f,who,k,inv,msb,off,m.start(),m.group()))
print(len(hits))
import collections
print(collections.Counter(h[1] for h in hits))
for h in hits[:60]: print(h)
