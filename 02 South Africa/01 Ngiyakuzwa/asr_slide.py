import torch, numpy as np, sys
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
src = sys.argv[1]; W, S = 12.0, 4.0
sr, x = wavfile.read(src); x = x.astype(np.float32)
for m in ["zionia/whisper-small-isizulu-noisy", "zionia/whisper-small-isizulu"]:
    proc = WhisperProcessor.from_pretrained(m); model = WhisperForConditionalGeneration.from_pretrained(m, torch_dtype=torch.float16).to("cuda")
    starts = np.arange(0, len(x)/sr - 2, S)
    segs = [x[int(a*sr):int((a+W)*sr)] for a in starts]
    segs = [s / (np.abs(s).max() + 1e-9) for s in segs]
    out = []
    for i in range(0, len(segs), 16):
        f = proc(segs[i:i+16], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        g = model.generate(f, num_beams=4, max_new_tokens=80, no_repeat_ngram_size=3)
        out += proc.batch_decode(g, skip_special_tokens=True)
    for a, t in zip(starts, out):
        print(f"{m[-5:]} {int(a//60):02d}:{a%60:04.1f} {t.strip()}")
