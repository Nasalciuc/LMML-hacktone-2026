import numpy as np, soundfile as sf, librosa, librosa.display, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A,B=305.5,307.3
fig,axs=plt.subplots(2,1,figsize=(13,7),sharex=True)
for ax,(tag,p) in zip(axs,[("clean","cleaned/voice_note_clean.wav"),("voiceband","cleaned/voice_note_clean_voiceband.wav")]):
    x,sr=sf.read(p); seg=x[int(A*sr):int(B*sr)].astype(np.float32)
    S=librosa.amplitude_to_db(np.abs(librosa.stft(seg,n_fft=1024,hop_length=128))+1e-6,ref=np.max)
    librosa.display.specshow(S,sr=sr,hop_length=128,x_axis="time",y_axis="linear",ax=ax,cmap="magma",vmin=-60,vmax=0)
    ax.set_title(f"{tag}  {A}-{B}s   (greedy: ikhodi~305.7-306.3, next word ~306.37-307.0)")
    ax.set_ylim(0,8000)
    for t in [305.7,306.3,306.4,306.65,306.8,307.0]:
        ax.axvline(t-A,color="cyan",lw=0.5,alpha=0.6)
plt.tight_layout(); plt.savefig("work/zoomword.png",dpi=110); print("ok")
