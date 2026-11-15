# Final ranking of the word after "ikhodi" (segment 067, 304.4-307.7 s) under 4 isiZulu models.
# Score = cost of the WORD only: NLL("<prefix> ikhodi W") - NLL("<prefix> ikhodi"), on the 3 cleaned
# versions (clean / strong / voiceband), two prefixes for Whisper. Reported raw (sum) and per char.
# Ranks are averaged over models. Lower = better. Writes work/decide_out.tsv.
import numpy as np, soundfile as sf, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from ctc_zul import load, logprobs, score

A, B = 304.4, 307.7
FILES = ["cleaned/voice_note_clean.wav", "cleaned/voice_note_clean_strong.wav", "cleaned/voice_note_clean_voiceband.wav"]
CANDS = ["isisho", "esisho", "isisi", "esisi", "imfihlo", "eyimfihlo", "ifihlo", "elisha", "elihle",
         "ifishu", "ifishi", "ifisho", "isifiso", "isifo", "ngisho", "isiswe", "ifu"]
REAL = {"isisho", "esisho", "imfihlo", "eyimfihlo", "ifihlo", "elisha", "elihle", "isifiso", "isifo", "ngisho", "isiswe", "ifu"}
segs = []
for f in FILES:
    x, sr = sf.read(f); s = x[int(A * sr):int(B * sr)].astype(np.float32); segs.append(s / np.abs(s).max())

res = {}  # model -> cand -> list of word costs
for key in ["mms", "w2vbert"]:
    proc, model = load(key); res[key] = {c: [] for c in CANDS}
    for s in segs:
        lp = logprobs(proc, model, s); pre = "kutshengisa ikhodi"; base = score(proc, lp, pre)
        for c in CANDS: res[key][c].append(score(proc, lp, f"{pre} {c}") - base)
    del model; torch.cuda.empty_cache()

for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    proc = WhisperProcessor.from_pretrained(m)
    model = WhisperForConditionalGeneration.from_pretrained(m).to("cuda").eval()
    key = "wh-" + ("noisy" if m.endswith("noisy") else "small"); res[key] = {c: [] for c in CANDS}
    def nll(feats, text):
        lab = proc.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad(): lg = model(input_features=feats, labels=lab).logits[0].float().log_softmax(-1)
        return -lg[torch.arange(lab.shape[1]), lab[0]][1:].sum().item()   # skip the forced start token
    for s in segs:
        feats = proc(s, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        for pre in ["kucingisa ikhodi", "kutshengisa ikhodi"]:
            base = nll(feats, pre)
            for c in CANDS: res[key][c].append(nll(feats, f"{pre} {c}") - base)
    del model; torch.cuda.empty_cache()

models = list(res)
mean = {k: {c: float(np.mean(v)) for c, v in res[k].items()} for k in models}
rank = {k: {c: r for r, c in enumerate(sorted(CANDS, key=lambda c: mean[k][c] / len(c)), 1)} for k in models}
avg = {c: np.mean([rank[k][c] for k in models]) for c in CANDS}
with open("work/decide_out.tsv", "w", encoding="utf-8") as fo:
    fo.write("word\treal\tavg_rank\t" + "\t".join(f"{k}_cost\t{k}_perchar\t{k}_rank" for k in models) + "\n")
    for c in sorted(CANDS, key=lambda c: avg[c]):
        fo.write(f"{c}\t{'Y' if c in REAL else '-'}\t{avg[c]:.2f}\t" + "\t".join(
            f"{mean[k][c]:.2f}\t{mean[k][c]/len(c):.2f}\t{rank[k][c]}" for k in models) + "\n")
print(open("work/decide_out.tsv", encoding="utf-8").read())
for k in models:
    rr = sorted(REAL, key=lambda c: mean[k][c] / len(c))
    print(f"{k:9s} real words best->: " + "  ".join(f"{c}:{mean[k][c]/len(c):.2f}" for c in rr[:5]))
