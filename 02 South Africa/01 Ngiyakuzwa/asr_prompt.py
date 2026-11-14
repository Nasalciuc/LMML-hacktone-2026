import glob, sys, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io import wavfile
import numpy as np
m = "openai/whisper-large-v3-turbo"
proc = WhisperProcessor.from_pretrained(m)
model = WhisperForConditionalGeneration.from_pretrained(m, torch_dtype=torch.float16).to("cuda")
prompt = "Sawubona. Ngiyabonga. Nansi ikhodi yethu. Ikhodi ithi:"
pids = torch.tensor(proc.get_prompt_ids(prompt)).to("cuda")
for pre in sys.argv[1].split(","):
    for f in sorted(glob.glob(f"segments/{pre}_*.wav")):
        sr, x = wavfile.read(f); x = x.astype(np.float32) / (32768 if x.dtype == np.int16 else 1)
        feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features.to("cuda", torch.float16)
        for lang in ["sw", "sn", "en"]:
            for nb in [1, 5]:
                ids = model.generate(feats, prompt_ids=pids, language=lang, num_beams=nb, max_new_tokens=60)
                t = proc.batch_decode(ids, skip_special_tokens=True)[0]
                print(pre, lang, nb, t.replace(prompt, "").strip())
