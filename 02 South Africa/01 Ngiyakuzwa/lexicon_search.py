# Lexicon-constrained search for the word after "ikhodi".
# For every word w in a ~30k isiZulu word list (+ hand candidates) compute the CTC likelihood of
# "ikhodi w" over 305.6-307.6 s, for each CTC model and each audio version. Per version the scores are
# shifted so the best word is 0, then summed over versions (lower = better). Output: lexicon_scores.tsv
import glob, sys, torch, numpy as np, soundfile as sf
import torch.nn.functional as F
from ctc_zul import load, logprobs
from score_ctc import CANDS

W0, W1 = 305.6, 307.6
VERSIONS = (["../task_south_african_republic/voice_note.wav:0", "cleaned/voice_note_clean.wav:0",
             "cleaned/voice_note_clean_strong.wav:0", "cleaned/voice_note_clean_voiceband.wav:0"]
            + [f.replace("\\", "/") + ":290" for f in sorted(glob.glob("work/enh/*.wav"))])

words = [l.split("\t")[0] for l in open("work/zulu_words.tsv", encoding="utf-8")]
words = list(dict.fromkeys(words + [c for c in CANDS if "-" not in c]))
keys = sys.argv[1:] or ["mms", "w2vbert"]


def word_scores(proc, lp, words, bs=1024):
    tok = proc.tokenizer
    blank = tok.pad_token_id
    T = lp.shape[0]
    out = []
    for i in range(0, len(words), bs):
        labs = [tok(f"ikhodi {w}").input_ids for w in words[i:i + bs]]
        L = torch.tensor([len(l) for l in labs])
        tgt = torch.tensor([t for l in labs for t in l])
        lpb = lp[:, None, :].expand(T, len(labs), lp.shape[1])
        loss = F.ctc_loss(lpb, tgt, torch.full((len(labs),), T), L, blank=blank, reduction="none", zero_infinity=False)
        out.append(loss.cpu())
    return torch.cat(out).numpy()


total = np.zeros(len(words))
per = {}
for key in keys:
    proc, model = load(key)
    for v in VERSIONS:
        f, off = v.rsplit(":", 1)
        x, sr = sf.read(f)
        # run the model on the whole phrase for context, then keep only the frames of "ikhodi X"
        a, b = 303.0, 309.0
        seg = x[int((a - float(off)) * sr):int((b - float(off)) * sr)]
        lp = logprobs(proc, model, seg)
        fr = lp.shape[0] / (b - a)
        lp = lp[int((W0 - a) * fr):int((W1 - a) * fr)]
        s = word_scores(proc, lp, words)
        s = np.where(np.isfinite(s), s, 1e4)
        s = s - s.min()
        name = f"{key}:{f.split('/')[-1].replace('.wav', '')}"
        per[name] = s
        total += np.minimum(s, 50)  # cap so one bad version can't dominate
        top = np.argsort(s)[:6]
        print(f"{name:42s} " + "  ".join(f"{words[j]}:{s[j]:.1f}" for j in top), flush=True)
    del model
    torch.cuda.empty_cache()

order = np.argsort(total)
print("\nSUM over versions (capped at 50 each), lower is better:")
for j in order[:25]:
    print(f"  {words[j]:16s} {total[j]:8.1f}")
with open("lexicon_scores.tsv", "w", encoding="utf-8") as fo:
    names = list(per)
    fo.write("word\ttotal\t" + "\t".join(names) + "\n")
    for j in order[:500]:
        fo.write(f"{words[j]}\t{total[j]:.1f}\t" + "\t".join(f"{per[n][j]:.1f}" for n in names) + "\n")
