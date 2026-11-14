import sys, soundfile as sf, numpy as np
from scipy import signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files=sys.argv[1].split(","); s,e=float(sys.argv[2]),float(sys.argv[3]); fmax=float(sys.argv[4])
fig,ax=plt.subplots(len(files),1,figsize=(20,5*len(files)),squeeze=False)
for k,fn in enumerate(files):
    x,sr=sf.read(fn); seg=x[int(s*sr):int(e*sr)]
    ff,tt,S=signal.stft(seg,sr,nperseg=1024,noverlap=960)
    L=20*np.log10(np.abs(S)+1e-9); a=ax[k,0]; a.pcolormesh(tt+s,ff,L,vmin=L.max()-60,vmax=L.max(),shading='auto',cmap='magma'); a.set_ylim(0,fmax)
plt.tight_layout(); plt.savefig(sys.argv[5],dpi=50)
