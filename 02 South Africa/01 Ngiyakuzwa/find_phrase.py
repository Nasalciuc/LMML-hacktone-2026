import glob, os, sys, json, numpy as np, torch
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
sys.stdout.reconfigure(encoding="utf-8")
CANDS = ["khona ikhodi lomgwaqo", "wentela uzosebenzisa ikhodi", "ikhodi engagunyaziwe elimazayo",
         "kutshengisa ikhono elihle", "ulwazi oluthile mayelana"]   # 4th = known control, 5th = a prompt that should match nothing special
files = sorted(glob.glob("segments/*.wav"))
N = {}
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    proc = WhisperProcessor.from_pretrained(m)
    model = WhisperForConditionalGeneration.from_pretrained(m).to("cuda").eval()
    labs = [proc.tokenizer(c, return_tensors="pt").input_ids.to("cuda") for c in CANDS]
    for f in files:
        sr, x = wavfile.read(f); x = x.astype(np.float32); x /= (np.abs(x).max() + 1e-9)
        feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
        with torch.no_grad():
            row = [model(input_features=feats, labels=lab).loss.item() * (lab.shape[1] - 1) for lab in labs]
        N.setdefault(os.path.basename(f)[:3], {})[m[-5:]] = row
    del model
segs = sorted(N)
for ci, c in enumerate(CANDS):
    tot = {k: N[k]["izulu"][ci] + N[k]["noisy"][ci] for k in segs}
    v = np.array([tot[k] for k in segs]); mu, sd = np.median(v), v.std()
    order = sorted(segs, key=lambda k: tot[k])
    print(f"== {c}   (NLL sum of 2 models; median {mu:.0f}, sd {sd:.0f})")
    print("   best: " + " | ".join(f"{k}:{tot[k]:.0f} (z {(tot[k]-mu)/sd:+.1f})" for k in order[:7]))
    print("   067: %.0f  077: %.0f  133: %.0f  004: %.0f  081: %.0f" % tuple(tot[k] for k in ["067", "077", "133", "004", "081"]))
json.dump(N, open("work/find_phrase_nll.json", "w"))
