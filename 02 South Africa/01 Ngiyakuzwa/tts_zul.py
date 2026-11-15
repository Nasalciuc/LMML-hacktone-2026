# Synthesize isiZulu text with facebook/mms-tts-zul. Usage: python tts_zul.py out.wav "text" [seed]
import sys, torch, numpy as np, soundfile as sf
from transformers import VitsModel, AutoTokenizer

M = "facebook/mms-tts-zul"
_tok = _mdl = None


def synth(text, seed=0, speed=1.0):
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained(M)
        _mdl = VitsModel.from_pretrained(M).eval()
    _mdl.speaking_rate = speed
    torch.manual_seed(seed)
    with torch.no_grad():
        y = _mdl(**_tok(text, return_tensors="pt")).waveform[0].numpy()
    return y.astype(np.float32), _mdl.config.sampling_rate


if __name__ == "__main__":
    y, sr = synth(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    sf.write(sys.argv[1], y, sr)
    print(sys.argv[1], sr, len(y) / sr)
