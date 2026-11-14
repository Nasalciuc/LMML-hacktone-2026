import sys, torch, collections, numpy as np
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
spans = [tuple(map(float, s.split(":"))) for s in sys.argv[1:]]
auds = {s: wavfile.read(f"cleaned/{s}.wav") for s in ["voice_note_clean", "voice_note_clean_voiceband", "voice_note_clean_strong"]}
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    proc = WhisperProcessor.from_pretrained(m); model = WhisperForConditionalGeneration.from_pretrained(m, torch_dtype=torch.float16).to("cuda")
    for a, b in spans:
        for s, (sr, x) in auds.items():
            x = x.astype(np.float32); seg = x[int(a*sr):int(b*sr)]; seg /= np.abs(seg).max() + 1e-9
            f = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            c = collections.Counter()
            for _ in range(6):
                g = model.generate(f, do_sample=True, temperature=0.7, num_return_sequences=8, max_new_tokens=40)
                c.update(t.strip() for t in proc.batch_decode(g, skip_special_tokens=True))
            print(f"{a:6.1f} {m[-5:]} {s[-9:]} |", c.most_common(6))
