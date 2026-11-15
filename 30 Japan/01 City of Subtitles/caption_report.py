"""Prints toorisugari's anonymous captions in posting order with line counts (-> out/captions_toorisugari.txt)."""
import json, csv, collections, os, sys
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
cap=json.load(open(os.path.join(OUT,'captions_ocr.json'),encoding='utf-8'))
tbl={r['file']:r['photographer'] for r in csv.DictReader(open(r"C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/attribution_table.csv",encoding='utf-8'))}
def who(f):
    if f.startswith('credited/'): return f.split('/')[1]
    return tbl.get(os.path.basename(f),'?')
# pools with stable index (sorted)
fh=set(); sh=set()
for f,ls in cap.items():
    for L in ls:
        if '、' in L:
            a,b=L.split('、',1); fh.add(a); sh.add(b)
        else: fh.add(L)
fh=sorted(fh); sh=sorted(sh)
fi={k:i for i,k in enumerate(fh)}; si={k:i for i,k in enumerate(sh)}
print("pool first=%d second=%d"%(len(fh),len(sh)))
# For each photographer, ordered files, dump line counts & first chars
for grp,ordered in [("ANON-TOORI", sorted([f for f in cap if f.startswith('anonymous/') and who(f)=='toorisugari']))]:
    print("\n=== %s (%d) ==="%(grp,len(ordered)))
    lc=[len(cap[f]) for f in ordered]
    print("linecounts:",lc)
    print("sum per week:")
    # first char of line1 (unicode)
    # second-half index of last line, etc -- dump raw
    for f in ordered:
        ls=cap[f]
        print(os.path.basename(f), len(ls), '||', ' @@ '.join(ls))
