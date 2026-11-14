import json
import sys

import torch
from transformers import pipeline

src = sys.argv[1] if len(sys.argv) > 1 else "cleaned/voice_note_clean.wav"
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16, device="cuda:0")
out = asr(src, chunk_length_s=10, batch_size=16, return_timestamps=True,
          generate_kwargs={"task": "transcribe", "temperature": 0.0})
rows = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"].strip()} for c in out["chunks"]]
json.dump(rows, open("transcript.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in rows:
    print(f"{r['start']:7.1f}-{r['end'] or 0:7.1f}  {r['text']}")
