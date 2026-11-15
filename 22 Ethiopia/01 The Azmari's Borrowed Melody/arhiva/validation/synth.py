import numpy as np
from scipy import signal

SR = 22050
QENET = {'tizita_maj': [0, 2, 4, 7, 9], 'tizita_min': [0, 2, 3, 7, 8], 'bati_maj': [0, 4, 5, 7, 11],
         'bati_min': [0, 3, 5, 7, 10], 'ambassel': [0, 1, 5, 7, 8], 'anchihoye': [0, 1, 5, 6, 9]}

def melody(rng, dur, tonic, scale, tempo_mean):
    degs = []
    cur = rng.integers(0, len(scale))
    notes = []
    t = 0.0
    while t < dur:
        step = rng.choice([-2, -1, -1, 0, 0, 1, 1, 2])
        cur = int(np.clip(cur + step, 0, len(scale)+1))
        octv, deg = divmod(cur, len(scale))
        midi = tonic + 12*octv + scale[deg]
        d = float(np.clip(rng.lognormal(np.log(tempo_mean), 0.5), 0.15, 2.0))
        notes.append((t, d, midi))
        t += d
    return notes

def body_gain(freqs, formants):
    g = np.zeros_like(freqs, dtype=float)
    for fc, bw, amp in formants:
        g += amp / (1 + ((freqs - fc)/bw)**2)
    return 0.2 + g

def make_take(rng, instrument, dur, tonic=None, scale=None, params=None):
    p = dict(params or {})
    tonic = tonic if tonic is not None else int(rng.integers(55, 72))
    scale = scale or QENET[rng.choice(list(QENET))]
    notes = melody(rng, dur + 1.0, tonic, scale, p.get('tempo', rng.uniform(0.35, 0.8)))
    n = int((dur + 2.5)*SR)
    y = np.zeros(n)
    if instrument == 'masenqo':
        formants = p.get('formants') or [(rng.uniform(250, 450), rng.uniform(60, 150), rng.uniform(1, 3)),
                                          (rng.uniform(700, 1100), rng.uniform(100, 300), rng.uniform(0.5, 2)),
                                          (rng.uniform(1500, 2500), rng.uniform(200, 500), rng.uniform(0.3, 1))]
        vib_rate = p.get('vib_rate', rng.uniform(4.5, 6.5)); vib_depth = p.get('vib_depth', rng.uniform(3, 15))
        nl = p.get('nl', rng.uniform(0.0, 0.15))
    else:
        vib_rate = p.get('vib_rate', rng.uniform(4.0, 6.0)); vib_depth = p.get('vib_depth', rng.uniform(5, 25))
        even_db = p.get('even_db', rng.uniform(-30, -12))
        breath = p.get('breath', rng.uniform(0.02, 0.08))
    tune = p.get('tune', rng.uniform(-30, 30))  # cents
    for (t0, d, midi) in notes:
        f0 = 440*2**((midi - 69 + tune/100)/12)
        att = 0.08 if instrument == 'masenqo' else 0.04
        rel = 0.12
        L = int((d + rel)*SR)
        tt = np.arange(L)/SR
        env = np.minimum(1, tt/att) * np.clip((d + rel - tt)/rel, 0, 1)
        vib = vib_depth/1200*np.log(2)*np.sin(2*np.pi*vib_rate*tt + rng.uniform(0, 6.28)) * np.minimum(1, tt/0.3)
        jit = np.cumsum(rng.normal(0, 1, L))/SR*0.3
        inst_f = f0*np.exp(vib + jit*0.001)
        phase = 2*np.pi*np.cumsum(inst_f)/SR
        s = np.zeros(L)
        K = int(min(40, (SR/2 - 200)/f0))
        if instrument == 'masenqo':
            ks = np.arange(1, K+1)
            amps = body_gain(ks*f0, formants)/ks
            for k, a in zip(ks, amps):
                s += a*np.sin(k*phase + 0.3*k)
            s = s/np.max(np.abs(s)+1e-9)
            s = s + nl*s**2
            noise = signal.lfilter(*signal.butter(2, [1000, 4000], btype='band', fs=SR), rng.normal(0, 0.01, L))
            s = s + noise
        else:
            ks = np.arange(1, K+1)
            amps = np.where(ks % 2 == 1, 1.0/ks, 10**(even_db/20)/ks)
            for k, a in zip(ks, amps):
                s += a*np.sin(k*phase)
            s = s/np.max(np.abs(s)+1e-9)
            trem = 1 + 0.15*signal.lfilter([1], [1, -0.995], rng.normal(0, 0.05, L))
            s = s*trem
            br = signal.lfilter(*signal.butter(2, [800, 6000], btype='band', fs=SR), rng.normal(0, 1, L))
            chiff = np.exp(-tt/0.03)
            s = s + br*(breath + 0.3*chiff)
            thump_f = rng.uniform(30, 100)
            s = s + 0.05*np.exp(-tt/0.3)*np.sin(2*np.pi*thump_f*tt)
        i0 = int(t0*SR)
        seg = s*env*p.get('amp', 1.0)
        y[i0:i0+L] += seg[:max(0, min(L, n-i0))]
    return y[:int(dur*SR)]

def chain(rng, y, cfg):
    out = y.copy()
    if cfg.get('hp'):
        out = signal.sosfilt(signal.butter(2, cfg['hp'], btype='high', fs=SR, output='sos'), out)
    if cfg.get('reverb'):
        rt = cfg['reverb']; L = int(rt*SR)
        h = rng.normal(0, 1, L)*np.exp(-6.9*np.arange(L)/L)
        h = h/np.sqrt(np.sum(h**2))
        wet = signal.fftconvolve(out, h)[:len(out)]
        out = out + cfg.get('wet', 0.3)*wet*np.std(out)/np.std(wet)
    rms = np.std(out)
    t = np.arange(len(out))/SR
    if cfg.get('hum'):
        fh = cfg.get('hum_f', 50.0) + 0.05*np.sin(2*np.pi*t/7)
        ph = 2*np.pi*np.cumsum(fh)/SR
        hum = sum(a*np.sin(k*ph) for k, a in [(1, 1), (2, 0.5), (3, 0.3)])
        out = out + rms*10**(cfg['hum']/20)*hum
    if cfg.get('noise'):
        out = out + rms*10**(cfg['noise']/20)*rng.normal(0, 1, len(out))
    if cfg.get('rumble'):
        r = signal.sosfilt(signal.butter(2, 30, fs=SR, output='sos'), rng.normal(0, 1, len(out)))
        out = out + rms*10**(cfg['rumble']/20)*r/np.std(r)
    if cfg.get('dc'):
        out = out + rms*cfg['dc']
    return out

def random_chain(rng):
    cfg = {}
    if rng.random() < 0.5: cfg['hp'] = float(rng.uniform(20, 120))
    if rng.random() < 0.5: cfg['reverb'] = float(rng.uniform(0.2, 1.2)); cfg['wet'] = float(rng.uniform(0.1, 0.5))
    if rng.random() < 0.5: cfg['hum'] = float(rng.uniform(-55, -35)); cfg['hum_f'] = float(rng.choice([50.0, 60.0]))
    if rng.random() < 0.6: cfg['noise'] = float(rng.uniform(-60, -35))
    if rng.random() < 0.3: cfg['rumble'] = float(rng.uniform(-45, -25))
    if rng.random() < 0.3: cfg['dc'] = float(rng.uniform(-0.01, 0.01))
    return cfg

def splice(base, ins, pos, fade):
    n = max(2, int(fade*SR)); L = len(ins)
    g = np.sqrt(np.mean(base[pos:pos+L]**2)/np.mean(ins**2))
    out = base.copy()
    w = np.ones(L); w[:n] = np.linspace(0, 1, n); w[-n:] = np.linspace(1, 0, n)
    out[pos:pos+L] = base[pos:pos+L]*(1-w) + ins*g*w
    return out

def make_case(seed, scenario):
    rng = np.random.default_rng(seed)
    total = rng.uniform(45, 62)
    L = rng.uniform(2.5, 10.0)
    instA = rng.choice(['masenqo', 'washint'])
    if scenario == 'diff_inst':
        instB = 'washint' if instA == 'masenqo' else 'masenqo'
        cA = random_chain(rng); cB = random_chain(rng)
        tA = make_take(rng, instA, total); tB = make_take(rng, instB, L + 2)
    elif scenario == 'same_inst_diff_chain':
        instB = instA
        cA = random_chain(rng); cB = random_chain(rng)
        tA = make_take(rng, instA, total); tB = make_take(rng, instB, L + 2)
    elif scenario == 'same_inst_same_chain':   # different performer params only
        instB = instA
        cA = random_chain(rng); cB = cA
        tA = make_take(rng, instA, total); tB = make_take(rng, instB, L + 2)
    A = chain(rng, tA, cA); B = chain(rng, tB, cB)
    st = rng.uniform(0, 1.5)
    ins = B[int(st*SR):int((st+L)*SR)]
    pos = int(rng.uniform(0.5, total - L - 0.5)*SR)
    y = splice(A, ins, pos, rng.choice([0.01, 0.03, 0.06, 0.12]))
    y = y/np.max(np.abs(y))*rng.uniform(1500, 12000)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16), pos/SR, (pos+len(ins))/SR, dict(instA=instA, instB=instB, cA=cA, cB=cB)

if __name__ == '__main__':
    import sys, os, json
    from scipy.io import wavfile
    n = int(sys.argv[1]); scen = sys.argv[2].split(',')
    d = '/home/claude/work/syn'
    rows = []; meta = {}
    for sc in scen:
        for s in range(n):
            y, a, b, m = make_case(hash((sc, s)) % (2**31), sc)
            fn = f'{d}/{sc}_{s:02d}.wav'
            wavfile.write(fn, SR, y)
            rows.append(f'{fn},{a:.3f},{b:.3f}'); meta[fn] = m
    open(f'{d}/truth.csv', 'w').write('\n'.join(rows)+'\n')
    json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)
    print(len(rows))
