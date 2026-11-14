import glob, os, sys, torch
from transformers import pipeline
model = sys.argv[1]
tag = model.split("/")[-1]
asr = pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device="cuda:0")
files = sorted(glob.glob("segments/*.wav"))
kw = {"num_beams": 5, "max_new_tokens": 120}
try:
    out = asr(files, batch_size=16, generate_kwargs=dict(kw, language="sw", task="transcribe"))
except Exception:
    out = asr(files, batch_size=16, generate_kwargs=kw)
with open(f"segments_{tag}.tsv", "w", encoding="utf-8") as f:
    for p, o in zip(files, out):
        f.write(f"{os.path.basename(p)[:3]}\t{o['text'].strip()}\n")
