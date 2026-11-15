# MMS-1b-all with the isiZulu adapter (CTC, letter output) on the phrase segments
# and on a few versions of the 05:04 phrase.
import glob, os, sys, torch, numpy as np, soundfile as sf
from transformers import Wav2Vec2ForCTC, AutoProcessor

M = "facebook/mms-1b-all"
proc = AutoProcessor.from_pretrained(M, target_lang="zul")
model = Wav2Vec2ForCTC.from_pretrained(M, target_lang="zul", ignore_mismatched_sizes=True).to("cuda").half().eval()


def rec(x):
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    inp = proc(x, sampling_rate=16000, return_tensors="pt").input_values.to("cuda").half()
    with torch.no_grad():
        lg = model(inp).logits[0].float()
    return proc.decode(lg.argmax(-1)), lg


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "seg"
    if mode == "seg":
        files = sorted(glob.glob("segments/*.wav"))
        with open("segments_mms-zul.tsv", "w", encoding="utf-8") as f:
            for p in files:
                x, sr = sf.read(p)
                t, _ = rec(x)
                f.write(f"{os.path.basename(p)[:3]}\t{t}\n")
                print(os.path.basename(p)[:3], t, flush=True)
    else:
        a, b = float(sys.argv[2]), float(sys.argv[3])
        for src in sys.argv[4:]:
            x, sr = sf.read(src)
            t, _ = rec(x[int(a * sr):int(b * sr)])
            print(f"{os.path.basename(src):40s} {t}")
