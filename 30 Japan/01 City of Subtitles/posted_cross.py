"""Posted-frame plain mean residual: toorisugari's anonymous photos (split by orientation so its sensor pattern
cannot line up) against the other regulars' anonymous photos (also split by orientation). A pattern shared by
everyone (processing) shows in both; one only in toorisugari's photos shows only in the first line."""
import torch

from render_sensor import info, d, names, dev


def mean_res(ix):
    s = sum(d["W"][i].float().to(dev) for i in ix)
    c = sum(d["M"][i].float().to(dev) for i in ix)
    return s / c.clamp(min=1)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


an = [i for i in range(len(names)) if names[i].startswith("anonymous")]
T0 = [i for i in an if info(i)[0] == "toorisugari" and info(i)[1] == 0]
T1 = [i for i in an if info(i)[0] == "toorisugari" and info(i)[1] == 1]
O0 = [i for i in an if info(i)[0] != "toorisugari" and info(i)[1] == 0]
O1 = [i for i in an if info(i)[0] != "toorisugari" and info(i)[1] == 1]
mT0, mT1, mO0, mO1 = map(mean_res, (T0, T1, O0, O1))
print("T0-T1", corr(mT0, mT1) * 1000, "O0-O1", corr(mO0, mO1) * 1000)
print("T0-O1", corr(mT0, mO1) * 1000, "T1-O0", corr(mT1, mO0) * 1000)
print("sizes", len(T0), len(T1), len(O0), len(O1))


def null_sd(a, b):
    sh = [(97, 131), (211, 353), (-157, 419), (331, -277), (-283, -199), (149, 557), (-401, 89), (53, -613),
          (389, 241), (-71, -463), (263, 29), (-337, 509)]
    import numpy as np
    return float(np.std([corr(a, torch.roll(b, s, (-2, -1))) for s in sh]))


print("null sd x1000:", null_sd(mT0, mT1) * 1000, null_sd(mT1, mO0) * 1000, null_sd(mO0, mO1) * 1000)
