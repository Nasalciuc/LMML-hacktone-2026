import glob, sys, torch
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda:0")
pre = sys.argv[1].split(",")
files = [f for p in pre for f in sorted(glob.glob(f"segments/{p}_*.wav")) + sorted(glob.glob(f"cleaned_seg/{p}_*.wav"))]
for lang in ["sn", "sw", "en", "ln", "ha", "yo"]:
    for beams in [1, 5]:
        out = asr(files, batch_size=8, generate_kwargs={"language": lang, "num_beams": beams, "temperature": 0.0})
        for f, o in zip(files, out):
            print(lang, beams, f.split("/")[-1][:3], o["text"].strip())
