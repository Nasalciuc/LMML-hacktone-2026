# Length-fair forced-alignment ranking of the word after "ikhodi".
# For each candidate w: CTC loss of full phrase "<pre> ikhodi w" minus CTC loss of the shared
# prefix "<pre> ikhodi" over the SAME phrase window -> isolates the word's alignment cost.
# Also reports that delta divided by #chars of w. Lower = better. Averaged over audio versions.
import sys, glob, torch, numpy as np, soundfile as sf
import torch.nn.functional as F
from ctc_zul import load, logprobs, score

A, B = 304.4, 307.7                       # phrase window (kutshengisa ikhodi X)
PRE  = "kutshengisa ikhodi"
# real isiZulu words that fit the acoustic skeleton + the literal greedy decodes as sanity refs
CANDS = ["isisho","esisho","imfihlo","eyimfihlo","ifihlo","elihle","elisha","isifiso","isifo",
         "isicelo","ngisho","okusho","elisho","ifishi","ifishu","isisi","inhliziyo","iphasiwedi",
         "ekisu","efisi","esishi","elishe","ekhaya","emuva","ifu"]

def ctc(lp, text):
    return score(proc, lp, text)

keys = sys.argv[1:] or ["mms","w2vbert"]
files = (["cleaned/voice_note_clean.wav:0","cleaned/voice_note_clean_strong.wav:0",
          "cleaned/voice_note_clean_voiceband.wav:0"] + [f.replace("\\","/")+":290" for f in sorted(glob.glob("work/enh/*.wav"))])
tot = {c:0.0 for c in CANDS}; totn = {c:0.0 for c in CANDS}; nver=0
perlines=[]
for key in keys:
    proc, model = load(key)
    for fo in files:
        f, off = fo.rsplit(":",1); off=float(off)
        x, sr = sf.read(f)
        seg = x[int((A-off)*sr):int((B-off)*sr)]
        if len(seg) < 100: continue
        lp = logprobs(proc, model, seg)
        base = ctc(lp, PRE)
        res=[]
        for c in CANDS:
            d = ctc(lp, f"{PRE} {c}") - base
            res.append((d, c))
        res.sort()
        mn = res[0][0]
        for d,c in res:
            tot[c]  += (d-mn)
            totn[c] += (d-mn)/len(c)
        nver+=1
        name=f"{key}:{f.split('/')[-1].replace('.wav','')}"
        perlines.append((name, [(c,d) for d,c in res]))
        print(f"{name:40s} " + "  ".join(f"{c}:{d:.1f}" for d,c in res[:6]), flush=True)
    del model; torch.cuda.empty_cache()

print(f"\n== SUM of (delta - best) over {nver} versions, lower=better ==")
for c,s in sorted(tot.items(), key=lambda t:t[1])[:15]:
    print(f"  {c:14s} {s:8.1f}   (per-char {totn[c]:6.2f})")
