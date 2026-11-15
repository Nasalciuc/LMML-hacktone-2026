# Neural enhancement of a window around the ikhodi phrase. Writes work/enh/<src>__<method>.wav
# (window starts at A seconds; ASR scripts add A back).
import os, sys, torch, numpy as np, soundfile as sf, julius

A, B = 290.0, 320.0
SRCS = {"raw": "../task_south_african_republic/voice_note.wav", "clean": "cleaned/voice_note_clean.wav"}
os.makedirs("work/enh", exist_ok=True)


def load(p):
    x, sr = sf.read(p)
    return torch.tensor(x[int(A * sr):int(B * sr)], dtype=torch.float32)


def save(name, y):
    y = y.detach().cpu().numpy().astype(np.float32).squeeze()
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    sf.write(f"work/enh/{name}.wav", y, 16000)
    print("wrote", name, flush=True)


def demucs_vocals(x):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model("htdemucs").cuda().eval()
    y = julius.resample_frac(x, 16000, m.samplerate)
    y = y[None, None].repeat(1, 2, 1).cuda()
    with torch.no_grad():
        out = apply_model(m, y, shifts=2, overlap=0.5)[0]
    v = out[m.sources.index("vocals")].mean(0)
    return julius.resample_frac(v.cpu(), m.samplerate, 16000)


def sb(x, kind):
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    if kind == "metricgan":
        m = SpectralMaskEnhancement.from_hparams("speechbrain/metricgan-plus-voicebank", savedir="work/sb_metricgan",
                                                 run_opts={"device": "cuda"})
        return m.enhance_batch(x[None].cuda(), lengths=torch.tensor([1.0]).cuda())[0]
    src = {"sepwham": "speechbrain/sepformer-wham16k-enhancement", "sepdns": "speechbrain/sepformer-dns4-16k-enhancement"}[kind]
    m = SepformerSeparation.from_hparams(src, savedir=f"work/sb_{kind}", run_opts={"device": "cuda"})
    outs = []
    hop = 16000 * 6
    for s in range(0, len(x), hop):  # 6 s chunks, keeps memory low
        with torch.no_grad():
            outs.append(m.separate_batch(x[s:s + hop][None].cuda())[0, :, 0].cpu())
    return torch.cat(outs)


if __name__ == "__main__":
    methods = sys.argv[1:] or ["demucs", "sepwham", "sepdns", "metricgan"]
    for sname, p in SRCS.items():
        x = load(p)
        for meth in methods:
            try:
                y = demucs_vocals(x) if meth == "demucs" else sb(x, meth)
                save(f"{sname}__{meth}", y)
            except Exception as e:
                print("FAIL", sname, meth, repr(e)[:300], flush=True)
            torch.cuda.empty_cache()
        # chains: demucs vocals then speech enhancement
    x = torch.tensor(sf.read("work/enh/raw__demucs.wav")[0], dtype=torch.float32) if os.path.exists("work/enh/raw__demucs.wav") else None
    if x is not None:
        for meth in ["sepdns", "metricgan"]:
            try:
                save(f"raw__demucs+{meth}", sb(x, meth))
            except Exception as e:
                print("FAIL chain", meth, repr(e)[:300])
