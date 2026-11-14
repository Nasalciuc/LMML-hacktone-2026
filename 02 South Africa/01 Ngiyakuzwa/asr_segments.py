import csv, glob, os, sys, torch
from transformers import pipeline
lang = sys.argv[1] if len(sys.argv) > 1 else "sw"
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda:0")
files = sorted(glob.glob("segments/*.wav"))
out = asr(files, batch_size=16, generate_kwargs={"language": lang, "task": "transcribe", "temperature": 0.0, "no_repeat_ngram_size": 3})
with open(f"segments_{lang}.tsv", "w", encoding="utf-8") as f:
    for p, o in zip(files, out):
        f.write(f"{os.path.basename(p)}\t{o['text'].strip()}\n")
