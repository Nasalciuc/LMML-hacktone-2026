import torch, numpy as np
d=torch.load("C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/cache/clean_res.pt",mmap=True)
names=d['names']; who=np.array(d['who']); rot=np.array(d['rot'])
R=d['R'].float().reshape(len(names),-1)
R=R/R.norm(dim=1,keepdim=True)
dev='cuda'
Rg=R.to(dev).half()
Rf=torch.flip(d['R'].float(),(-2,-1)).reshape(len(names),-1); Rf=(Rf/Rf.norm(dim=1,keepdim=True)).to(dev).half()
Cp=(Rg@Rg.T).float().cpu().numpy()
Cr=(Rg@Rf.T).float().cpu().numpy()
np.savez('gram2.npz',Cp=Cp,Cr=Cr,names=np.array(names),who=who,rot=rot)
N=len(names); iu=np.triu_indices(N,1)
diff=who[iu[0]]!=who[iu[1]]
sd_p=Cp[iu][diff].std(); sd_r=Cr[iu][diff].std()
print('null sd posted %.2e rot %.2e'%(sd_p,sd_r))
H6=sorted(set(who))
for h in H6:
    ix=np.where(who==h)[0]
    for nm,C,sd in (('posted',Cp,sd_p),('rot',Cr,sd_r)):
        sub=C[np.ix_(ix,ix)]/sd
        same=rot[ix][:,None]==rot[ix][None,:]
        m=~np.eye(len(ix),dtype=bool)
        print(h,nm,'same-rot mean z %.2f  opp-rot mean z %.2f'%(sub[m&same].mean(),sub[m&~same].mean()))
