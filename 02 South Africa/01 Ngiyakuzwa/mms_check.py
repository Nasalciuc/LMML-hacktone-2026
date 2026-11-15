import sys, numpy as np, torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, AutoProcessor
sys.stdout.reconfigure(encoding="utf-8")
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="zul")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="zul", ignore_mismatched_sizes=True).to("cuda").eval()
for src in ["voice_note_clean", "voice_note_clean_strong", "voice_note_clean_voiceband"]:
    sr, x = wavfile.read(f"cleaned/{src}.wav"); x = x.astype(np.float32); x /= np.abs(x).max()
    for a, b in [(434.78, 440.75), (436.5, 441.0), (438.5, 441.2), (304.5, 307.8), (347.9, 350.4)]:
        seg = x[int(a * sr):int(b * sr)]
        inp = proc(seg, sampling_rate=sr, return_tensors="pt").to("cuda")
        with torch.no_grad(): ids = model(**inp).logits.argmax(-1)[0]
        print(f"{src[11:]:16s} {a:.1f}-{b:.1f}: {proc.decode(ids)}")
