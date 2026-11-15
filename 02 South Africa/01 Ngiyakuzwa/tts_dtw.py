# Independent check. facebook/mms-tts-zul does NOT exist on HF, so a related Bantu MMS voice is used
# (tso = Xitsonga default, or swh); spelling is phonetic in both. Synthesize candidate phrases and DTW-match (MFCC) against the
# real audio. Phrase-level ("ikhodi X", 305.55-307.15) and word-only ("X", 306.25-307.1).
# Lower normalized DTW = better. Prints incrementally.
import sys, numpy as np, soundfile as sf, librosa, torch
from transformers import VitsModel, AutoTokenizer
CANDS = ["isisho","esisho","imfihlo","eyimfihlo","ifishu","ifishi","elisha","elihle",
         "isifiso","isicelo","ngisho","isifo","ifu"]
SR=16000
def mfcc(y):
    y=librosa.util.normalize(y.astype(np.float32))
    m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    f=np.vstack([m,librosa.feature.delta(m)])
    return (f-f.mean(1,keepdims=True))/(f.std(1,keepdims=True)+1e-6)
def dtwd(a,b):
    D,wp=librosa.sequence.dtw(X=a,Y=b,metric="euclidean"); return D[-1,-1]/len(wp)
TTS=sys.argv[1] if len(sys.argv)>1 else "facebook/mms-tts-tso"
tok=AutoTokenizer.from_pretrained(TTS)
mdl=VitsModel.from_pretrained(TTS).eval()
def synth(text,seed,speed):
    mdl.speaking_rate=speed; torch.manual_seed(seed)
    with torch.no_grad(): y=mdl(**tok(text,return_tensors="pt")).waveform[0].numpy()
    sr=mdl.config.sampling_rate
    return librosa.resample(y,orig_sr=sr,target_sr=SR) if sr!=SR else y
def rw(path,a,b):
    x,sr=sf.read(path)
    if sr!=SR: x=librosa.resample(x.astype(np.float32),orig_sr=sr,target_sr=SR)
    return x[int(a*SR):int(b*SR)]
REF={}
for tag,p in [("clean","cleaned/voice_note_clean.wav"),("vband","cleaned/voice_note_clean_voiceband.wav")]:
    REF[(tag,"phrase")]=mfcc(rw(p,305.55,307.15)); REF[(tag,"word")]=mfcc(rw(p,306.25,307.10))
rows={}
for c in CANDS:
    best={}
    for mode,txt in [("phrase",f"ikhodi {c}"),("word",c)]:
        ds=[]
        for seed in (0,1):
            for sp in (0.8,1.0):
                y=synth(txt,seed,sp)
                if mode=="word": y=y[-int(0.85*SR):] if len(y)>int(0.85*SR) else y
                fy=mfcc(y)
                ds += [dtwd(REF[(t,mode)],fy) for t in ("clean","vband")]
        best[mode]=min(ds)
    rows[c]=best
    print(f"{c:12s} phrase={best['phrase']:.3f}  word={best['word']:.3f}", flush=True)
print("\n== ranked by phrase DTW ==",flush=True)
for c,b in sorted(rows.items(),key=lambda t:t[1]['phrase']): print(f"  {c:12s} {b['phrase']:.3f}",flush=True)
print("\n== ranked by word-only DTW ==",flush=True)
for c,b in sorted(rows.items(),key=lambda t:t[1]['word']): print(f"  {c:12s} {b['word']:.3f}",flush=True)
