"""Does camera B's fingerprint match another camera's fingerprint after a flip / 180-degree turn / shift?"""
import torch

from render_fp import members, fp

cams = ["toorisugari_B", "toorisugari", "yakin", "amayadori", "jihanki", "rojiura", "shuuden"]
K = {}
for c in cams:
    k, den = fp(members(c))
    K[c] = (k - k.mean()) / k.std()
ops = {"id": lambda k: k, "rot180": lambda k: torch.flip(k, (0, 1)), "flipH": lambda k: torch.flip(k, (1,)),
       "flipV": lambda k: torch.flip(k, (0,))}
for a in cams[:2]:
    for b in cams:
        for nm, op in ops.items():
            if a == b and nm == "id":
                continue
            kb = op(K[b])
            xc = torch.fft.irfft2(torch.fft.rfft2(K[a]) * torch.conj(torch.fft.rfft2(kb)), s=K[a].shape)
            e = (xc ** 2).mean()
            i = int(torch.argmax(xc))
            H, W = xc.shape
            py, px = divmod(i, W)
            print(f"{a:14s} vs {nm:6s}({b:13s}) pce0 {float(xc[0,0]**2/e):7.1f}  peak ({(py+H//2)%H-H//2:5d},{(px+W//2)%W-W//2:5d}) pce {float(xc.flatten()[i]**2/e):7.1f}", flush=True)
