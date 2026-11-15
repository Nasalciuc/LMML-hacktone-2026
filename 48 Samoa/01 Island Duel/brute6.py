import json, math, itertools
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]; guides=[s for s in stars if s["size"]>6]; comps=[s for s in stars if s["size"]<=6]
def dist(a,b): return math.hypot(a["x"]-b["x"],a["y"]-b["y"])
assign={}
for c in comps: assign.setdefault(id(min(guides,key=lambda g:dist(c,g))),[]).append(c)
top=min(guides,key=lambda s:s["y"]); bottom=max(guides,key=lambda s:s["y"])
left=min(guides,key=lambda s:s["x"]); right=max(guides,key=lambda s:s["x"]); faint=min(guides,key=lambda s:s["size"])
geo=[bottom,left,top,right,faint]
# build token list: (letter, greek(1..5), rank(1..), globalpos(1..))
toks=[]; p=0
for gi,g in enumerate(geo,1):
    for r,c in enumerate(sorted(assign[id(g)],key=lambda c:dist(c,g)),1):
        p+=1; toks.append((c["text"],gi,r,p))
letters=[t[0] for t in toks]
print("tokens:", [(t[0],t[1],t[2],t[3]) for t in toks])

SAM=set("AEFGILMNOPSTUV")
THEME=("STAR HOME WARD NAVIGAT VOYAG CROSS SOUTH WIND WAVE OCEAN ISLAND PATH WAYFIND "
 "POLAR GUIDE LIGHT NIGHT SAIL MOON COMPASS CHART HORIZON CELEST MARINER VASA FETU "
 "SAMOA PACIFIC APIA FOLAU MALAGA TAUTAI TAPUITEA MATAGI AVANOA FETUAO RETURN DAWN "
 "SOUTHERN STARPATH HOMEWARD VOYAGE").split()
BIG={'TH':3.6,'HE':3.1,'IN':2.4,'ER':2.1,'AN':2.0,'RE':1.7,'ON':1.6,'AT':1.5,'EN':1.4,'ND':1.4,
'TI':1.4,'ES':1.4,'OR':1.3,'TE':1.3,'ED':1.3,'IS':1.3,'IT':1.2,'AL':1.2,'AR':1.2,'ST':1.2,'TO':1.2,
'NT':1.1,'NG':1.1,'SE':1.1,'HA':1.1,'AS':1.1,'OU':1.0,'IO':1.0,'LE':1.0,'VE':1.0,'CO':.9,'ME':.9,
'DE':.9,'HI':.9,'RI':.9,'RO':.9,'IC':.8,'NE':.8,'EA':.8,'RA':.8,'CE':.8,'LI':.8,'LL':.7,'MA':.7,
'SI':.7,'UR':.7,'CA':.6,'EL':.6,'TA':.6,'LA':.6,'NA':.6,'SO':.6,'HO':.6,'AU':.6,'AI':.6,'OA':.5}
def eng(s):
    sc=sum(BIG.get(s[i:i+2],0) for i in range(len(s)-1))
    for w in THEME:
        if w in s and len(w)>=4: sc+=len(w)*2.5
    v=sum(c in "AEIOU" for c in s)/len(s)
    sc+=-5 if (v<0.2 or v>0.72) else 0
    return sc
def sam(s):
    if any(c not in SAM for c in s): return -1
    v=sum(c in "AEIOU" for c in s)/len(s)
    return 10+v*10  # all-samoan + vowel-rich

best_e={}; best_s=[]
for A in range(26):
 for B in range(26):
  for C in range(26):
   for D in range(26):
    for sign in (-1,1):
     out=[]
     for (ch,g,r,pp) in toks:
         sh=(A*g+B*r+C*pp+D)%26
         out.append(chr((ord(ch)-65+sign*sh)%26+65))
     s="".join(out)
     e=eng(s)
     if s not in best_e or e>best_e[s][0]:
         best_e[s]=(e,f"A{A}B{B}C{C}D{D}s{sign}")
     sc=sam(s)
     if sc>14:
         best_s.append((sc,f"A{A}B{B}C{C}D{D}s{sign}",s))
ranked=sorted(best_e.items(),key=lambda kv:-kv[1][0])
print("\nTOP 30 English-like:")
for s,(e,nm) in ranked[:30]: print(f"  {e:5.1f} {nm:16s} {s}")
best_s=sorted(set(best_s),reverse=True)
print(f"\nAll-Samoan hits (vowel-rich), top 30 of {len(best_s)}:")
for sc,nm,s in best_s[:30]: print(f"  {sc:4.1f} {nm:16s} {s}")
