import json, math, itertools
W = r"C:/Users/user/Projects/LMML-hacktone-2026/17 Cuba/01 If you know you know/index/words.txt"
DICT = set()
for line in open(W, encoding="utf-8", errors="ignore"):
    t = line.split()
    if t:
        w = t[0].strip().upper()
        if w.isalpha() and len(w) >= 3:
            DICT.add(w)

d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]; guides=[s for s in stars if s["size"]>6]; comps=[s for s in stars if s["size"]<=6]
def dist(a,b): return math.hypot(a["x"]-b["x"],a["y"]-b["y"])
assign={}
for c in comps: assign.setdefault(id(min(guides,key=lambda g:dist(c,g))),[]).append(c)
def grp(order_g, withg=False, far=False):
    seq=[]
    for g in order_g:
        if withg: seq.append(g["text"])
        for c in sorted(assign[id(g)],key=lambda c:dist(c,g),reverse=far): seq.append(c["text"])
    return "".join(seq)

orders={}
for perm in itertools.permutations(guides):
    tag="".join(g["text"] for g in perm)
    orders["c_"+tag]=grp(perm)
    orders["cf_"+tag]=grp(perm,far=True)
    orders["g_"+tag]=grp(perm,withg=True)
orders["yx"]="".join(c["text"] for c in sorted(comps,key=lambda c:(c["y"],c["x"])))
orders["xy"]="".join(c["text"] for c in sorted(comps,key=lambda c:(c["x"],c["y"])))
orders["yxa"]="".join(s["text"] for s in sorted(stars,key=lambda c:(c["y"],c["x"])))
for k in list(orders): orders[k+"R"]=orders[k][::-1]

def caesar(t,k): return "".join(chr((ord(c)-65-k)%26+26)%26 and chr((ord(c)-65-k)%26+65) for c in t)
def caesar(t,k): return "".join(chr((ord(c)-65-k)%26+65) for c in t)
def vig(t,key,sign): return "".join(chr((ord(c)-65+sign*(ord(key[i%len(key)])-65))%26+65) for i,c in enumerate(t))
def prog(t,a,b,sign): return "".join(chr((ord(c)-65+sign*(a+b*i))%26+65) for i,c in enumerate(t))

guide_letters=[g["text"] for g in guides]
KEYS=set("".join(p) for p in itertools.permutations(guide_letters))
KEYS|={"SIGMOID","SAMOA","HOMEWARD","HOME","STAR","FETU","CROSS","SOUTH","GREEK",
       "OVER","WATCH","COMPASS","POLARIS","NIGHT","VOYAGE","FOUNDATION","CHART","FIVE"}

def words_in(s):
    # longest dictionary word that is a substring, plus exact-match bonus
    best="";
    n=len(s)
    for i in range(n):
        for j in range(i+4, n+1):
            sub=s[i:j]
            if sub in DICT and len(sub)>len(best): best=sub
    full = s in DICT
    return best, full

hits=[]
def consider(nm,s):
    best,full=words_in(s)
    val=(100 if full else 0)+len(best)
    if val>=5 or full:
        hits.append((val, nm, s, best, full))

for onm,t in orders.items():
    for k in range(26): consider(f"{onm}|C{k}", caesar(t,k))
    for key in KEYS:
        consider(f"{onm}|Vd-{key}", vig(t,key,-1))
        consider(f"{onm}|Ve-{key}", vig(t,key,1))
    for a in range(26):
        for b in range(26):
            consider(f"{onm}|P{a},{b}-", prog(t,a,b,-1))
            consider(f"{onm}|P{a},{b}+", prog(t,a,b,1))

hits.sort(reverse=True)
seen=set(); n=0
print("TOP dictionary hits (val=100+len if exact word, else longest substring word):")
for val,nm,s,best,full in hits:
    if s in seen: continue
    seen.add(s)
    print(f"  val={val:4d} {'EXACT' if full else '     '} {nm:22s} {s}   [{best}]")
    n+=1
    if n>=50: break
