import torch, numpy as np, sys
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
a, b = 303.5, 308.5
cands = ["kucingisa ikhodi imfihlo", "kucingisa ikhodi iyimfihlo", "kucingisa ikhodi ifishi", "kucingisa ikhodi isisho",
         "kucingisa ikhodi isisi", "kucingisa ikhodi ifish", "qinisekisa ikhodi imfihlo", "kugcinwa ikhodi iyimfihlo",
         "kutshengiswa ikhodi imfihlo", "kutshengiswa ikhodi isisho", "kucingisa ikhodi ifiso", "kucingisa ikhodi isifo",
         "kucingisa ikhodi ifasitela", "kucingisa ikhodi imvula", "kucingisa ikhodi ifisi", "kucingisa ikhodi isiswe", "kucingisa ikhodi ifishu"]
for m in ["zionia/whisper-small-isizulu", "zionia/whisper-small-isizulu-noisy"]:
    proc = WhisperProcessor.from_pretrained(m); model = WhisperForConditionalGeneration.from_pretrained(m).to("cuda").eval()
    for src in ["voice_note_clean", "voice_note_clean_voiceband", "voice_note_clean_strong"]:
        sr, x = wavfile.read(f"cleaned/{src}.wav"); seg = x.astype(np.float32)[int(a*sr):int(b*sr)]; seg /= np.abs(seg).max()
        f = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        res = []
        for c in cands:
            lab = proc.tokenizer(c, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = model(input_features=f, labels=lab)
            res.append((out.loss.item() * (lab.shape[1] - 1), c))
        res.sort()
        print(m[-5:], src[-9:], " | ".join(f"{c.split()[-1]}:{s:.1f}" for s, c in res[:8]))
