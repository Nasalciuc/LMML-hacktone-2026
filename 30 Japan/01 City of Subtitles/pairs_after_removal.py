"""After removing toorisugari's own sensor term from each of its photos, do any of them still share a pattern
(a second camera hidden in a subset)? Pairwise z of the cleaned residuals, in both relative orientations."""
import numpy as np
import torch

from render_sensor import info, d, names, dev, rot, sensor_fp

h = "toorisugari"
K, _, _ = sensor_fp(h)
ix = [i for i in range(len(names)) if info(i)[0] == h]
R = []
for i in ix:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    k = rot(K) if info(i)[1] else K
    s = k * li
    w = w - (w * s).sum() / (s * s).sum() * s
    x = w * li / li.pow(2).mean().sqrt()
    x = x - x.mean()
    R.append((x / x.norm()).half())
R = torch.stack(R).flatten(1)
Rr = torch.stack([rot(r.view(3, 960, 1280)).flatten() for r in R])
G0 = (R.float() @ R.float().T).cpu().numpy()
G1 = (R.float() @ Rr.float().T).cpu().numpy()
n = len(ix)
off = G0[~np.eye(n, dtype=bool)]
sd = 1.4826 * np.median(np.abs(off - np.median(off)))
Z0, Z1 = G0 / sd, G1 / sd
np.fill_diagonal(Z0, 0)
print("sd", sd, "max z same frame", Z0.max(), "max z turned", Z1.max())
pairs = sorted([(Z0[a, b], "same", a, b) for a in range(n) for b in range(a + 1, n)] +
               [(Z1[a, b], "turn", a, b) for a in range(n) for b in range(n) if a != b], reverse=True)[:25]
for z, k, a, b in pairs:
    print(f"{z:6.2f} {k} {names[ix[a]]} {names[ix[b]]}")
w, v = np.linalg.eigh(np.clip(Z0, -3, 10))
print("top eigenvalues same-frame", np.round(w[-5:], 1))
