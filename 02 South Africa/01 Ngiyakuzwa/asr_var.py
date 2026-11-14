import sys, torch, collections, numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration
a, b = float(sys.argv[1]), float(sys.argv[2])
srcs = ["cleaned/voice_note_clean.wav", "cleaned/voice_note_clean_strong.wav", "cleaned/voice_note_clean_voiceband.wav", "../task_south_african_republic/voice_note.wav"]
import glob, os
srcs = [s for s in srcs if os.path.exists(s)] + glob.glob("../task_*/*.wav")[:1]
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy", "openai/whisper-large-v3-turbo"]:
    proc = WhisperProcessor.from_pretrained(m); model = WhisperForConditionalGeneration.from_pretrained(m, torch_dtype=torch.float16).to("cuda")
    for s in srcs:
        sr, x = wavfile.read(s); x = x.astype(np.float32); x /= np.abs(x).max() + 1e-9
        seg = x[int(a*sr):int(b*sr)]
        if sr != 16000: seg = resample_poly(seg, 16000, sr)
        f = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        kw = dict(language="sw") if "large" in m else {}
        g = model.generate(f, num_beams=5, max_new_tokens=40, **kw)
        best = proc.batch_decode(g, skip_special_tokens=True)[0]
        c = collections.Counter()
        for _ in range(4):
            g = model.generate(f, do_sample=True, temperature=0.7, num_return_sequences=8, max_new_tokens=40, **kw)
            c.update(t.strip() for t in proc.batch_decode(g, skip_special_tokens=True))
        print(m.split("/")[-1][:22], os.path.basename(s)[:24], "|", best, "|", c.most_common(5))
