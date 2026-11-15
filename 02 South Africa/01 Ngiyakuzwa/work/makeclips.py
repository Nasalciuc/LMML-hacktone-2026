import numpy as np, soundfile as sf, librosa, os
OUT=r"C:/Users/user/Downloads/LMML_upload/South Africa/asculta"
SR=16000
def cut(path,a,b):
    x,sr=sf.read(path)
    if sr!=SR: x=librosa.resample(x.astype(np.float32),orig_sr=sr,target_sr=SR)
    return x[int(a*SR):int(b*SR)].astype(np.float32)
def norm(y): return (y/(np.abs(y).max()+1e-9)*0.97).astype(np.float32)
jobs={
 "ikhodi_WORD_clean":     ("cleaned/voice_note_clean.wav",     305.55,307.20),
 "WORD_only_clean":       ("cleaned/voice_note_clean.wav",     306.25,307.15),
 "WORD_only_voiceband":   ("cleaned/voice_note_clean_voiceband.wav",306.25,307.15),
 "WORD_only_strong":      ("cleaned/voice_note_clean_strong.wav",306.25,307.15),
 "ikhodi_WORD_raw":       ("../task_south_african_republic/voice_note.wav",305.55,307.20),
}
for name,(p,a,b) in jobs.items():
    y=norm(cut(p,a,b))
    sf.write(os.path.join(OUT,name+".wav"),y,SR)
    ys=librosa.effects.time_stretch(y,rate=0.75); ys=norm(ys)
    sf.write(os.path.join(OUT,name+"_0.75x.wav"),ys,SR)
    print(name, f"{b-a:.2f}s")
print("done ->", OUT)
