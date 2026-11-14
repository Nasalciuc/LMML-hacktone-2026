import sys, soundfile as sf, numpy as np
from scipy import signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a,sr=sf.read(sys.argv[1]); b,_=sf.read(sys.argv[2]); s,e=float(sys.argv[3]),float(sys.argv[4])
fig,ax=plt.subplots(2,1,figsize=(18,9))
for k,x in enumerate((a,b)):
    seg=x[int(s*sr):int(e*sr)]
    ff,tt,S=signal.stft(seg,sr,nperseg=512,noverlap=448)
    L=20*np.log10(np.abs(S)+1e-9); ax[k].pcolormesh(tt+s,ff,L,vmin=L.max()-70,vmax=L.max(),shading='auto',cmap='magma'); ax[k].set_ylim(0,8000)
plt.tight_layout(); plt.savefig(sys.argv[5],dpi=55)
