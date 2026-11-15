"""Is something in toorisugari's ANONYMOUS photos that is not in his credited ones? Fingerprints estimated
separately (luma residual, MLE sum(W*I)/sum(I^2)): K_c from the 18 credited photos, K_a from the 45
anonymous ones (or one week's 9), each photo turned to the sensor frame by its rotation flag.
Sensor frame: D = K_a - a*K_c (a by least squares).
Posted frame: K_a built without turning; D = K_a_posted - a0*K_c - a1*rot180(K_c) (the two orientations of the
sensor fitted jointly). D is multiplied by sqrt(sum I^2) so its noise level is flat over the frame
(otherwise rows covered by fewer photos, near the captions, dominate). Renders out/ksplit_<who>_<scope>_<frame>.png: 6 panels = raw, Gaussian 2/4/8, |D|
blurred 4, local energy (16 px box) thresholded at its 90th percentile. Yakin is the control."""
import csv, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from scipy import ndimage
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
D0 = torch.load(HERE / "cache" / "clean_res.pt"); names, who, rot = D0["names"], D0["who"], D0["rot"]
flip = lambda t: torch.flip(t, (0, 1))
def K(ix, sensor=True):
    num = 0; den = 0
    for i in ix:
        w = d["W"][i].float().mean(0); I = d["I"][i].float().mean(0); m = d["M"][i].float()
        if sensor and rot[i]: w, I, m = flip(w), flip(I), flip(m)
        num = num + w * I * m; den = den + I * I * m
    k = num / den.clamp(min=1); k = k - k.mean(1, keepdim=True); return (k - k.mean(0, keepdim=True)), den
def fitsub(y, xs, w):
    A = torch.stack([x[w] for x in xs], 1); c = torch.linalg.lstsq(A, y[w][:, None]).solution[:, 0]
    return y - sum(ci * x for ci, x in zip(c, xs)), [round(float(v), 3) for v in c]
def norm(a, lo=1, hi=99):
    p1, p2 = np.percentile(a, [lo, hi]); return np.clip((a - p1) / (p2 - p1 + 1e-12) * 255, 0, 255).astype(np.uint8)
def sheet(Dm, valid, tag, info):
    a = Dm.numpy() * valid
    pan = [("raw", norm(a)), ("gauss2", norm(ndimage.gaussian_filter(a, 2))), ("gauss4", norm(ndimage.gaussian_filter(a, 4))),
           ("gauss8", norm(ndimage.gaussian_filter(a, 8))), ("abs_blur4", norm(ndimage.gaussian_filter(np.abs(a), 4))),
           ("energy16>p90", ((e := ndimage.uniform_filter(a * a, 16)) > np.percentile(e[valid > 0], 90)).astype(np.uint8) * 255)]
    W, H = 640, 480; S = Image.new("L", (W * 3, (H + 18) * 2), 0); dr = ImageDraw.Draw(S)
    for k, (n, im) in enumerate(pan):
        x, y = (k % 3) * W, (k // 3) * (H + 18); S.paste(Image.fromarray(im).resize((W, H)), (x, y + 18)); dr.text((x + 4, y + 3), f"{tag} {n} {info}", fill=255)
    S.save(HERE / "out" / f"ksplit_{tag}.png")
    Image.fromarray(pan[3][1]).save(HERE / "out" / f"ksplit_{tag}_gauss8_full.png")
    Image.fromarray(norm(ndimage.uniform_filter(a * a, 16))).save(HERE / "out" / f"ksplit_{tag}_energy16_full.png")
anon = lambda h: sorted([i for i in range(len(names)) if who[i] == h and names[i].startswith("anonymous")], key=lambda i: names[i])
cred = lambda h: [i for i in range(len(names)) if who[i] == h and not names[i].startswith("anonymous")]
for h in ("toorisugari", "yakin"):
    Kc, dc = K(cred(h))
    scopes = [("all", anon(h))] + ([(f"w{w:02d}", [i for i in anon(h) if names[i].startswith(f"anonymous/w{w:02d}")]) for w in range(8, 13)] if h == "toorisugari" else [])
    for sc, ix in scopes:
        Ka, da = K(ix); v = ((da > 0) & (dc > 0)).float()
        Ds, cs = fitsub(Ka, [Kc], v > 0)
        r = float(torch.corrcoef(torch.stack([Ka[v > 0], Kc[v > 0]]))[0, 1])
        Ds = Ds * da.clamp(min=0).sqrt()  # whiten: MLE noise sd ~ 1/sqrt(sum I^2)
        sheet(Ds, v.numpy(), f"{h}_{sc}_sensor", f"n={len(ix)} corr(Ka,Kc)={r:.3f} a={cs}")
        Kp, dp = K(ix, sensor=False); vp = ((dp > 0) & (dc > 0) & (flip(dc) > 0)).float()
        Dp, cp = fitsub(Kp, [Kc, flip(Kc)], vp > 0)
        Dp = Dp * dp.clamp(min=0).sqrt()
        if True:
            sheet(Dp, vp.numpy(), f"{h}_{sc}_posted", f"n={len(ix)} a0,a1={cp}")
        print(f"{h:11s} {sc:4s} n={len(ix):2d} corr(Ka,Kc)={r:.4f} sensor a={cs} posted a0,a1={cp}", flush=True)
