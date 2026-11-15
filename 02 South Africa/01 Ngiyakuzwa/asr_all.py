# Free transcription of the ikhodi phrase (302.5-309.5 s) with four isiZulu models on every audio version.
# Output: asr_ikhodi_phrase.tsv (model, version, text).
import glob, torch, numpy as np, soundfile as sf
from transformers import pipeline
from ctc_zul import load, logprobs, greedy_times

A, B = 302.5, 309.5
FILES = (["../task_south_african_republic/voice_note.wav:0", "cleaned/voice_note_clean.wav:0",
          "cleaned/voice_note_clean_strong.wav:0", "cleaned/voice_note_clean_voiceband.wav:0"]
         + [f.replace("\\", "/") + ":290" for f in sorted(glob.glob("work/enh/*.wav"))])


def cut(fo):
    f, off = fo.rsplit(":", 1)
    x, sr = sf.read(f)
    s = x[int((A - float(off)) * sr):int((B - float(off)) * sr)].astype(np.float32)
    return f.split("/")[-1].replace(".wav", ""), s / (np.abs(s).max() + 1e-9)


segs = [cut(f) for f in FILES]
rows = []
for key in ["mms", "w2vbert"]:
    proc, model = load(key)
    for name, s in segs:
        g = greedy_times(proc, logprobs(proc, model, s), B - A, A)
        rows.append((key, name, "".join(c for c, _ in g).replace("|", " ").strip()))
        print(*rows[-1], sep="\t", flush=True)
    del model; torch.cuda.empty_cache()
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    asr = pipeline("automatic-speech-recognition", model=m, torch_dtype=torch.float16, device="cuda:0")
    for name, s in segs:
        o = asr(s.copy(), generate_kwargs={"num_beams": 5, "max_new_tokens": 60})
        rows.append((m.split("/")[-1].replace("whisper-small-", "wsm-"), name, o["text"].strip()))
        print(*rows[-1], sep="\t", flush=True)
    del asr; torch.cuda.empty_cache()
with open("asr_ikhodi_phrase.tsv", "w", encoding="utf-8") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")
