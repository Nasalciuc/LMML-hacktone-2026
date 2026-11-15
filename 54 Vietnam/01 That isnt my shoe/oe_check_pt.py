"""Fresh-process check of the exported file. Imports torch/torchvision only (+ numpy/PIL to read
photos), never the training code. CPU only.

python oe_check_pt.py vietnam_model.pt
"""
import csv, os, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401  (the platform has it; the file itself does not need it)
from PIL import Image

path = sys.argv[1]
TASK = Path(__file__).resolve().parent.parent / "task_vietnam"
torch.set_num_threads(4)
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| file", path,
      f"{os.path.getsize(path)} bytes = {os.path.getsize(path)/1e6:.2f} MB ({os.path.getsize(path)/2**20:.2f} MiB)")

m1 = torch.jit.load(path, map_location="cpu")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m2 = torch.load(path, weights_only=False, map_location="cpu")
print("torch.jit.load ->", type(m1).__name__, "| torch.load(weights_only=False) ->", type(m2).__name__)
print("classes", m1.classes, "| class_to_idx", m1.class_to_idx, "| score", m1.score_name, "| thr", m1.thr, "| input_size", m1.input_size)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.load(path)
    print("torch.load(path) with the default weights_only=True: OK")
except Exception as e:
    print("torch.load(path) with the default weights_only=True: fails, as for every TorchScript file ->", str(e).split("\n")[0][:80])

rows = list(csv.DictReader(open(TASK / "train-output.csv")))[::7][:100]
imgs = [np.asarray(Image.open(TASK / "train-input" / r["filename"]).convert("RGB")) for r in rows]
labels = [r["label"] for r in rows]
full = torch.from_numpy(np.stack(imgs))                      # B,H,W,3 uint8
crop = full[:, 26:230, 26:230].contiguous()                  # centre crops, 204x204: the file has to resize them
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def variants(hwc):
    chw = hwc.permute(0, 3, 1, 2).contiguous()
    f01 = chw.float() / 255
    small = torch.nn.functional.interpolate(f01, size=224, mode="bilinear", antialias=True)
    return {
        "uint8 B,H,W,3": hwc,
        "float 0-1 B,3,H,W": f01,
        "float 0-255 B,3,H,W": chw.float(),
        "float 0-1 B,H,W,3": hwc.float() / 255,
        "float64 0-1 B,3,H,W": chw.double() / 255,
        "float16 0-1 B,3,H,W": f01.half(),
        "ImageNet-normalised B,3,H,W": (f01 - MEAN) / STD,
        "(x-0.5)/0.5 B,3,H,W": (f01 - 0.5) / 0.5,
        "RGBA uint8 B,4,H,W": torch.cat([chw, torch.full((len(chw), 1) + tuple(chw.shape[2:]), 255, dtype=torch.uint8)], 1),
        "float 0-1, caller resized to 224": small,
        "ImageNet-norm., caller resized to 224": (small - MEAN) / STD,
    }


for name, base in [("100 training photos, 256x256", full), ("their 204x204 centre crops", crop)]:
    ref = m1.predict(base.permute(0, 3, 1, 2).contiguous())            # reference: uint8 B,3,H,W
    print(f"-- {name}: reference = uint8 B,3,H,W; accuracy {np.mean([a == b for a, b in zip(ref, labels)]):.2f}, "
          f"labelled unknown {np.mean([a == 'unknown' for a in ref]):.2f}")
    for vn, x in variants(base).items():
        p = m2.predict(x)
        print(f"   {vn:38s} same answer as reference {np.mean([a == b for a, b in zip(p, ref)]):.2f}")
    one_hwc = [m1.predict(base[i])[0] for i in range(10)]
    one_chw = [m1.predict(base[i].permute(2, 0, 1).float() / 255)[0] for i in range(10)]
    print("   single (H,W,3) uint8 / single (3,H,W) float: same answer",
          np.mean([a == b for a, b in zip(one_hwc, ref[:10])]), "/", np.mean([a == b for a, b in zip(one_chw, ref[:10])]))

chw = full.permute(0, 3, 1, 2).contiguous()
print("grayscale (B,1,H,W) runs:", m1.predict(chw.float().mean(1, keepdim=True)[:3]))
print("non-square 300x400 runs:", m1.predict(torch.from_numpy(np.asarray(Image.fromarray(imgs[0]).resize((400, 300))))))
print("random noise / flat grey:", m1.predict(torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)), m1.predict(torch.full((1, 3, 256, 256), 128, dtype=torch.uint8)))
t = time.time(); out = m1(chw); dt = time.time() - t
print("forward ->", tuple(out.shape), out.dtype, "| rows sum to", round(float(out.sum(1).mean()), 4),
      f"| min {float(out.min()):.4f} max {float(out.max()):.4f} | {1000*dt/len(chw):.0f} ms/image on CPU (4 threads)")
print("argmax(forward) == predict:", [m1.classes[i] for i in out.argmax(1).tolist()] == m1.predict(chw),
      "| predict_index:", m1.predict_index(chw[:5]).tolist())
lg, s = m1.logits_and_score(chw[:3])
print("logits_and_score ->", tuple(lg.shape), tuple(s.shape), [round(v, 3) for v in s.tolist()])
