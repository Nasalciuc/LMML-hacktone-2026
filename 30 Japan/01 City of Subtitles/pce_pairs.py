"""Step 2: pairwise PCE between cleaned residuals (cache/clean_res.pt), all 253 photos, full circular shift
search via FFT cross-correlation. Relative transforms of photo B: id, rot180, flipLR, flipUD on the full
960x1280 frame; rot90, rot270 on the 960x960 centre crop. PCE = peak^2 / mean(c^2 outside 11x11 around the
peak), peak = largest |c| (sign kept). Output cache/pce_pairs.npz: per variant PCE, signed peak, dy, dx.
With argument "phat": phase-only correlation (cross-spectrum divided by its magnitude), which suppresses
scene texture and repetitive structure; output cache/pce_pairs_phat.npz."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = torch.load(HERE / "cache" / "clean_res.pt")
R = D["R"].to(dev)  # float16 on GPU, ~620 MB
N = R.shape[0]
VAR = {
    "id": (lambda t: t, False), "rot180": (lambda t: torch.flip(t, (-2, -1)), False),
    "flipLR": (lambda t: torch.flip(t, (-1,)), False), "flipUD": (lambda t: torch.flip(t, (-2,)), False),
    "rot90": (lambda t: torch.rot90(t[..., :, 160:1120], 1, (-2, -1)), True),
    "rot270": (lambda t: torch.rot90(t[..., :, 160:1120], 3, (-2, -1)), True),
}
def pce_batch(c):
    B, H, W = c.shape
    flat = c.reshape(B, -1); idx = flat.abs().argmax(1); pk = flat.gather(1, idx[:, None])[:, 0]
    py, px = idx // W, idx % W
    tot = (flat * flat).sum(1)
    oy = torch.arange(-5, 6, device=c.device); yy = (py[:, None] + oy) % H; xx = (px[:, None] + oy) % W
    win = c[torch.arange(B, device=c.device)[:, None, None], yy[:, :, None], xx[:, None, :]]
    e = (tot - (win * win).sum((1, 2))) / (H * W - 121)
    dy = torch.where(py > H // 2, py - H, py); dx = torch.where(px > W // 2, px - W, px)
    return (pk * pk / e), pk, dy, dx
PHAT = len(sys.argv) > 1 and sys.argv[1] == "phat"
OUTF = HERE / "cache" / ("pce_pairs_phat.npz" if PHAT else "pce_pairs.npz")
out = {}
BS = 12
for name, (tf, sq) in VAR.items():
    P = np.zeros((N, N), np.float32); S = np.zeros((N, N), np.float32); DY = np.zeros((N, N), np.int16); DX = np.zeros((N, N), np.int16)
    for i in range(N):
        a = R[i].float(); a = a[:, 160:1120] if sq else a
        FA = torch.fft.rfft2(a)
        js = list(range(i + 1, N))
        for k in range(0, len(js), BS):
            jj = js[k:k + BS]
            b = tf(R[jj].float())
            X = FA[None] * torch.conj(torch.fft.rfft2(b))
            if PHAT: X = X / X.abs().clamp(min=1e-12)
            c = torch.fft.irfft2(X, s=a.shape)
            p, pk, dy, dx = pce_batch(c)
            P[i, jj] = p.cpu().numpy(); S[i, jj] = pk.cpu().numpy(); DY[i, jj] = dy.cpu().numpy(); DX[i, jj] = dx.cpu().numpy()
    out[name + "_pce"] = P; out[name + "_peak"] = S; out[name + "_dy"] = DY; out[name + "_dx"] = DX
    iu = np.triu_indices(N, 1); v = P[iu]
    print(f"{name:7s} pairs {v.size}  median PCE {np.median(v):.1f}  99.9% {np.percentile(v, 99.9):.1f}  max {v.max():.1f}  >60: {(v > 60).sum()}", flush=True)
    np.savez_compressed(OUTF, **out)
print("done")
