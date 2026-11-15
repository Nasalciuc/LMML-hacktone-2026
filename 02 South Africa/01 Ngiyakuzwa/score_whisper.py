# Teacher-forced log-likelihood of "<prefix> ikhodi <word>" under the isiZulu Whisper fine-tunes.
import sys, glob, torch, numpy as np, soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from score_ctc import CANDS, A, B

PREFIXES = ["kucingisa ikhodi", "kutshengisa ikhodi"]
files = sys.argv[1:] or (["cleaned/voice_note_clean.wav:0", "cleaned/voice_note_clean_strong.wav:0",
                          "cleaned/voice_note_clean_voiceband.wav:0"] + [f + ":290" for f in sorted(glob.glob("work/enh/*.wav"))])
cands = list(dict.fromkeys(c.replace("-", " ") for c in CANDS))
tot = {c: 0.0 for c in cands}
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    proc = WhisperProcessor.from_pretrained(m)
    model = WhisperForConditionalGeneration.from_pretrained(m).to("cuda").eval()
    for fo in files:
        f, off = fo.rsplit(":", 1)
        x, sr = sf.read(f)
        seg = x[int((A - float(off)) * sr):int((B - float(off)) * sr)].astype(np.float32)
        seg /= np.abs(seg).max()
        feats = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        for pre in PREFIXES:
            res = []
            for c in cands:
                lab = proc.tokenizer(f"{pre} {c}", return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    out = model(input_features=feats, labels=lab)
                res.append((out.loss.item() * (lab.shape[1] - 1), c))
            res.sort()
            best = res[0][0]
            for s, c in res:
                tot[c] += s - best
            print(f"{m[-5:]} {f.split('/')[-1][:30]:30s} {pre[:5]} " + "  ".join(f"{c}:{s - best:.1f}" for s, c in res[:6]), flush=True)
print("\nSUM of (score - best), lower is better:")
for c, s in sorted(tot.items(), key=lambda t: t[1])[:15]:
    print(f"  {c:16s} {s:8.1f}")
