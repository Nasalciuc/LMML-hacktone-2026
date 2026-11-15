import json, csv, collections, os, sys
OUT=sys.argv[1]
cap=json.load(open(os.path.join(OUT,'captions.json'),encoding='utf-8'))
tbl={r['file']:r['photographer'] for r in csv.DictReader(open(r"C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/attribution_table.csv",encoding='utf-8'))}
# distinct phrase pool
alllines=collections.Counter()
firsthalf=collections.Counter(); secondhalf=collections.Counter()
for f,lines in cap.items():
    for L in lines:
        alllines[L]+=1
        if '、' in L:
            a,b=L.split('、',1); firsthalf[a]+=1; secondhalf[b]+=1
print("total distinct lines:",len(alllines))
print("distinct first-halves:",len(firsthalf),"distinct second-halves:",len(secondhalf))
print("\n--- most common full lines ---")
for L,c in alllines.most_common(15): print(c,L)
# toorisugari anon ordered
anon_toori=sorted([f for f in cap if f.startswith('anonymous/') and tbl.get(os.path.basename(f))=='toorisugari'])
print("\n=== Toorisugari anon captions in order ===")
for f in anon_toori:
    print(os.path.basename(f), '|', ' / '.join(cap[f]))
