import sys
sys.path.insert(0, '/home/claude/work/final')
import numpy as np
from scipy import signal
from scipy.io import wavfile
import main as M

sr, d = wavfile.read('/home/claude/data/train-input.wav')
x0 = d.astype(np.float64)
T0, T1 = 14.960, 20.965
rng = np.random.default_rng(0)
def run(name, x, s=sr):
    a, b = M.detect(x, s)
    ok = abs(a-T0) <= 1.5 and abs(b-T1) <= 1.5
    print(f'{name:28s} {a:6.2f} {b:6.2f}  {"OK" if ok else "FAIL"}')
run('orig', x0)
for lvl in (5, 20, 60):
    run(f'white noise sd={lvl}', x0 + rng.normal(0, lvl, len(x0)))
for fc in (20, 80, 150, 300):
    run(f'highpass {fc}', signal.sosfiltfilt(signal.butter(4, fc, btype='high', fs=sr, output='sos'), x0))
for fc in (3000, 5000):
    run(f'lowpass {fc}', signal.sosfiltfilt(signal.butter(6, fc, fs=sr, output='sos'), x0))
run('bandpass 200-4000', signal.sosfiltfilt(signal.butter(4, [200, 4000], btype='band', fs=sr, output='sos'), x0))
for s2 in (16000, 44100, 48000, 8000):
    run(f'resample {s2}', signal.resample_poly(x0, s2//50, sr//50) if s2 % 50 == 0 else signal.resample(x0, int(len(x0)*s2/sr)), s2)
run('gain x10', x0*10)
run('gain x0.05', np.round(x0*0.05))
run('hum 50Hz added', x0 + 30*np.sin(2*np.pi*50*np.arange(len(x0))/sr))
run('crop 10..50', x0[int(10*sr):int(50*sr)])  # truth shifts by -10 -> will show FAIL label but check values
