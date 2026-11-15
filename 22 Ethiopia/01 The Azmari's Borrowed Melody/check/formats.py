"""Runs main.py the way the grader does, on the train recording saved in many formats.

    python formats.py <train-input.wav> <empty work folder> [main.py to test]

For each variant: copies main.py into the work folder, writes the recording
there as test-1-input.wav, runs `python main.py test-1-input.wav` in that
folder and reads detection.csv. The work folder is wiped first. The last
lines feed it silence, a 0.2 s clip and a file that is not audio: it has to
exit 0 and still write a valid CSV.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

train, work = sys.argv[1], sys.argv[2]
main_py = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
sr, xi = wavfile.read(train)
x = xi.astype(np.float64)
T = (14.96, 20.965)


def variants():
    yield "int16 as given", sr, xi, T
    yield "float32 -1..1", sr, (x / 32768).astype(np.float32), T
    yield "float64", sr, x / 32768, T
    yield "int32 (scaled up)", sr, (x * 65536).astype(np.int32), T
    yield "uint8 8-bit", sr, np.clip(x / 256 * 20 + 128, 0, 255).astype(np.uint8), T
    yield "stereo (same both sides)", sr, np.stack([xi, xi], 1), T
    yield "stereo (right silent)", sr, np.stack([xi, np.zeros_like(xi)], 1), T
    yield "44.1 kHz", 44100, signal.resample_poly(x, 2, 1).astype(np.int16), T
    yield "16 kHz", 16000, signal.resample_poly(x, 320, 441).astype(np.int16), T
    yield "8 kHz", 8000, signal.resample_poly(x, 160, 441).astype(np.int16), T
    yield "10x louder (near full scale)", sr, np.clip(x * 20, -32768, 32767).astype(np.int16), T
    yield "10x quieter", sr, (x / 10).astype(np.int16), T
    yield "DC offset +500", sr, (x + 500).astype(np.int16), T
    yield "polarity inverted", sr, (-x).astype(np.int16), T
    yield "first 30 s only", sr, xi[: 30 * sr], T
    yield "12-24 s only", sr, xi[12 * sr: 24 * sr], (2.96, 8.965)
    yield "3 s of silence", sr, np.zeros(3 * sr, np.int16), None
    yield "0.2 s clip", sr, xi[: int(0.2 * sr)], None


shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
shutil.copy(main_py, os.path.join(work, "main.py"))
fn = "test-1-input.wav"
out = os.path.join(work, "detection.csv")
for name, rate, data, truth in variants():
    wavfile.write(os.path.join(work, fn), rate, data)
    if os.path.exists(out):
        os.remove(out)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "main.py", fn], cwd=work, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    rows = open(out).read().split() if os.path.exists(out) else []
    res = rows[1] if len(rows) > 1 else "NO CSV"
    verdict = ""
    if truth and len(rows) > 1:
        s, e = map(float, res.split(","))
        verdict = "ok" if abs(s - truth[0]) <= 1.5 and abs(e - truth[1]) <= 1.5 else "WRONG"
    print(f"{name:30s} exit {p.returncode}  {dt:4.1f} s  header {rows[0] if rows else '-':13s} -> {res:16s} {verdict}")

with open(os.path.join(work, fn), "wb") as fh:
    fh.write(b"this is not audio")
os.remove(out)
p = subprocess.run([sys.executable, "main.py", fn], cwd=work, capture_output=True, text=True)
rows = open(out).read().split() if os.path.exists(out) else ["-", "NO CSV"]
extra = sorted(set(os.listdir(work)) - {"main.py", fn, "detection.csv"})
print(f"{'not an audio file':30s} exit {p.returncode}          header {rows[0]:13s} -> {rows[1]:16s} extra files written: {extra}")
