"""Step 2b: scaled variants. Each of toorisugari's 63 photos (A) against all 253 photos (B), B resized by
0.90 / 0.95 / 1.05 / 1.10 (bilinear, centre crop or zero pad back to 960x1280), relative orientation id and
rot180, full shift search, phase-only (PHAT) correlation, PCE as in pce_pairs.py. Prints the top pairs per scale; cache/pce_scales.npz."""
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from pce_pairs_lib import pce_batch
HERE = Path(__file__).resolve().parent
dev = "cuda"
D = torch.load(HERE / "cache" / "clean_res.pt")
R = D["R"].to(dev); names, who = D["names"], D["who"]; N = R.shape[0]
A = [i for i in range(N) if who[i] == "toorisugari"]
def rs(b, s):
    H, W = b.shape[-2:]; h, w = round(H * s), round(W * s)
    x = F.interpolate(b[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    out = torch.zeros(b.shape[0], H, W, device=b.device)
    if s >= 1:
        y0, x0 = (h - H) // 2, (w - W) // 2; out = x[:, y0:y0 + H, x0:x0 + W]
    else:
        y0, x0 = (H - h) // 2, (W - w) // 2; out[:, y0:y0 + h, x0:x0 + w] = x
    return out
res = {}
for s in (0.90, 0.95, 1.05, 1.10):
    for vn, tf in (("id", lambda t: t), ("rot180", lambda t: torch.flip(t, (-2, -1)))):
        P = np.zeros((len(A), N), np.float32)
        for a_k, i in enumerate(A):
            FA = torch.fft.rfft2(R[i].float())
            for k in range(0, N, 12):
                jj = [j for j in range(k, min(N, k + 12))]
                b = tf(rs(R[jj].float(), s))
                X = FA[None] * torch.conj(torch.fft.rfft2(b)); X = X / X.abs().clamp(min=1e-12)
                c = torch.fft.irfft2(X, s=(960, 1280))
                P[a_k, jj] = pce_batch(c)[0].cpu().numpy()
            P[a_k, i] = 0
        res[f"s{s}_{vn}"] = P
        t = np.argsort(-P.ravel())[:5]
        print(f"scale {s} {vn:6s} median {np.median(P):.1f} max {P.max():.1f} >60: {(P > 60).sum()}  top: " +
              ", ".join(f"{names[A[x // N]]}~{names[x % N]} {P.ravel()[x]:.0f}" for x in t), flush=True)
np.savez_compressed(HERE / "cache" / "pce_scales.npz", **res)
