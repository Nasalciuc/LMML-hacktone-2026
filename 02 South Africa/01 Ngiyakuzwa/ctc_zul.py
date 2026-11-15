# CTC isiZulu models: greedy letters with times, and forced-alignment (CTC loss) scores of candidate words.
# python ctc_zul.py <model> <a> <b> <wav> [cand1 cand2 ...]   (candidates are scored as "ikhodi <cand>")
import sys, torch, numpy as np, soundfile as sf
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ForCTC

MODELS = {"mms": "facebook/mms-1b-all", "w2vbert": "kesbeast23/isizulu-w2v-bert-2.0"}


def load(key):
    m = MODELS.get(key, key)
    if "mms" in m:
        proc = AutoProcessor.from_pretrained(m, target_lang="zul")
        model = Wav2Vec2ForCTC.from_pretrained(m, target_lang="zul", ignore_mismatched_sizes=True)
    else:
        proc = AutoProcessor.from_pretrained(m)
        model = AutoModelForCTC.from_pretrained(m)
    return proc, model.to("cuda").half().eval()


def logprobs(proc, model, x):
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    inp = proc(x, sampling_rate=16000, return_tensors="pt")
    inp = {k: (v.to("cuda").half() if v.dtype.is_floating_point else v.to("cuda")) for k, v in inp.items()}
    with torch.no_grad():
        lg = model(**inp).logits[0].float()
    return lg.log_softmax(-1)


def greedy_times(proc, lp, dur, t0):
    ids = lp.argmax(-1).tolist()
    step = dur / len(ids)
    blank = proc.tokenizer.pad_token_id
    out, prev = [], None
    for i, k in enumerate(ids):
        if k != prev and k != blank:
            out.append((proc.tokenizer.convert_ids_to_tokens(k), t0 + i * step))
        prev = k
    return out


def score(proc, lp, text):
    lab = proc.tokenizer(text).input_ids
    lab = torch.tensor([lab], device=lp.device)
    l = F.ctc_loss(lp[:, None, :], lab, torch.tensor([lp.shape[0]]), torch.tensor([lab.shape[1]]),
                   blank=proc.tokenizer.pad_token_id, reduction="sum", zero_infinity=True)
    return l.item()


if __name__ == "__main__":
    key, a, b, wav = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    proc, model = load(key)
    x, sr = sf.read(wav)
    seg = x[int(a * sr):int(b * sr)]
    lp = logprobs(proc, model, seg)
    g = greedy_times(proc, lp, b - a, a)
    print("".join(c for c, _ in g).replace("|", " "))
    print(" ".join(f"{c}{t:.2f}" for c, t in g))
