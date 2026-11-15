# Word timestamps from the isiZulu Whisper fine-tunes on a span. python asr_words.py a b src...
import sys, torch, numpy as np, soundfile as sf
from transformers import pipeline

a, b = float(sys.argv[1]), float(sys.argv[2])
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    asr = pipeline("automatic-speech-recognition", model=m, torch_dtype=torch.float16, device="cuda:0")
    for src in sys.argv[3:]:
        x, sr = sf.read(src)
        seg = x[int(a * sr):int(b * sr)].astype(np.float32)
        seg /= np.abs(seg).max()
        o = asr(seg.copy(), return_timestamps="word", generate_kwargs={"num_beams": 1})
        w = " ".join(f"{c['text'].strip()}[{a + c['timestamp'][0]:.2f}-{a + (c['timestamp'][1] or 0):.2f}]" for c in o["chunks"])
        print(m.split("-")[-1][:6], src.split("\\")[-1][:28], "|", w, flush=True)
