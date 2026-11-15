import glob, os, sys, numpy as np, torch
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
sys.stdout.reconfigure(encoding="utf-8")
models = {}
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    models[m[-5:]] = (WhisperProcessor.from_pretrained(m), WhisperForConditionalGeneration.from_pretrained(m).to("cuda").eval())
srcs = {k: wavfile.read(f"cleaned/{k}.wav") for k in ["voice_note_clean", "voice_note_clean_strong", "voice_note_clean_voiceband"]}
def nll(mk, src, a, b, text):
    proc, model = models[mk]; sr, x = srcs[src]
    seg = x[int(a * sr):int(b * sr)].astype(np.float32); seg /= (np.abs(seg).max() + 1e-9)
    f = proc(seg, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
    lab = proc.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        return model(input_features=f, labels=lab).loss.item() * (lab.shape[1] - 1)
PRE = "wentela uzosebenzisa ikhodi"
words = ["ulwazi", "ukwazi", "uyazi", "umlazi", "ulwandle", "uluwazi", "ucwazi", "ulazi", "ulwasi", "ubuntu", "imfihlo", "isisho", "elihle",
         "lomgwaqo", "umsebenzi", "uhulumeni", "abantu", "isizulu", "imali", "impilo", "ulimi", "izwe", "amanzi", "umuzi", "inkosi", "ulwazi oluthile", "ngolwazi", "nolwazi", "lolwazi", "yolwazi"]
a, b = 434.78, 440.75
print(f"window {a}-{b}: cost of the extra word after '{PRE}' (NLL(PRE + w) - NLL(PRE)), summed over 2 models x 3 cleanings; lower = better")
base = {(mk, s): nll(mk, s, a, b, PRE) for mk in models for s in srcs}
rows = []
for w in words:
    per = [nll(mk, s, a, b, PRE + " " + w) - base[(mk, s)] for mk in models for s in srcs]
    rows.append((sum(per), w, per))
rows.sort()
for tot, w, per in rows:
    print(f"  {tot:7.1f}  {w:18s} " + " ".join(f"{p:5.1f}" for p in per))
# is the 4-word phrase specific to this segment?
import csv
idx = list(csv.DictReader(open("segments/index.csv", encoding="utf-8")))
full = PRE + " ulwazi"
tot = []
for r in idx:
    a2, b2 = float(r["start_s"]), float(r["end_s"])
    tot.append((sum(nll(mk, "voice_note_clean", a2, b2, full) for mk in models), r["n"]))
v = np.array([t for t, _ in tot]); tot.sort()
print(f"\n'{full}' over all 138 segments: median {np.median(v):.0f} sd {v.std():.0f}; best: " + " | ".join(f"seg {n}: {t:.0f} (z {(t-np.median(v))/v.std():+.1f})" for t, n in tot[:5]))
