# Forced-alignment scores: CTC negative log-likelihood of "<prefix> ikhodi <word>" over the phrase,
# for many candidate words, on several audio versions. Lower = better. Reports gap to the runner-up.
# python score_ctc.py <model-key> [wav[:offset] ...]
import sys, glob, torch, numpy as np, soundfile as sf
from ctc_zul import load, logprobs, score

CANDS = """imfihlo iyimfihlo ifihlo imfihlakalo elihle lihle elisha isisho ifishi ifisho ifishu efushu isifiso isifo
isihlalo isihlahla imfene imfologo imfundo imfuyo imfucuza ifulethi isiswe inhlanhla insimbi imbali imvula ifu ilifu
ibhola ihlobo ihlelo ifasitela inhliziyo ilitshe ibhishi isibuko inyoni ilanga inkanyezi ifusi imfishi imfeshi
isisi esisi enisu insizwa insipho insizi iphasiwedi iphasiwede imfihlo-yami ilihle ehlile isiphi isifishane imfushane
umfula ifoni ifiso imvelo imfene isilwane isihluthulelo ufudu ingwe indlovu imbuzi inkosi isizulu""".split()

A, B = 304.6, 307.6          # phrase "kucingisa ikhodi X" (from word times)
PREFIXES = ["kutshengisa ikhodi", "ikhodi"]


def windows(path, off):
    x, sr = sf.read(path)
    return x[int((A - off) * sr):int((B - off) * sr)]


if __name__ == "__main__":
    key = sys.argv[1]
    files = sys.argv[2:] or (["cleaned/voice_note_clean.wav:0", "cleaned/voice_note_clean_strong.wav:0",
                              "cleaned/voice_note_clean_voiceband.wav:0"] + [f + ":290" for f in sorted(glob.glob("work/enh/*.wav"))])
    proc, model = load(key)
    cands = list(dict.fromkeys(c.replace("-", " ") for c in CANDS))
    tot = {c: 0.0 for c in cands}
    for fo in files:
        f, off = fo.rsplit(":", 1)
        seg = windows(f, float(off))
        lp = logprobs(proc, model, seg)
        for pre in PREFIXES:
            if pre == "ikhodi":  # word-only window: ikhodi + next word
                s0 = int((305.6 - A) / (B - A) * lp.shape[0])
                l = lp[s0:]
            else:
                l = lp
            r = sorted((score(proc, l, f"{pre} {c}"), c) for c in cands)
            best = r[0][0]
            for s, c in r:
                tot[c] += s - best
            print(f"{f.split('/')[-1][:34]:34s} {pre[:5]:5s} " + "  ".join(f"{c}:{s - best:.1f}" for s, c in r[:6]), flush=True)
    print("\nSUM of (score - best) over all versions, lower is better:")
    for c, s in sorted(tot.items(), key=lambda t: t[1])[:15]:
        print(f"  {c:16s} {s:8.1f}")
