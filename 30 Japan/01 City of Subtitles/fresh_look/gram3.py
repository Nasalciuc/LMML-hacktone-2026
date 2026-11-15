import numpy as np
z=np.load('gram2.npz'); Cp,Cr,names,who,rot=z['Cp'],z['Cr'],z['names'],z['who'],z['rot']
N=len(names); iu=np.triu_indices(N,1); diff=who[iu[0]]!=who[iu[1]]
Zp=Cp/Cp[iu][diff].std(); Zr=Cr/Cr[iu][diff].std()
def analyse(h,anon_only=False,verbose=False):
    ix=np.where(who==h)[0]
    if anon_only: ix=np.array([i for i in ix if names[i].startswith('anonymous')])
    r=rot[ix]; same=r[:,None]==r[None,:]
    out={}
    for geo in ('posted','sensor','antisensor'):
        if geo=='posted': M=Zp[np.ix_(ix,ix)].copy()
        elif geo=='sensor': M=np.where(same,Zp[np.ix_(ix,ix)],Zr[np.ix_(ix,ix)])
        else: M=np.where(same,Zr[np.ix_(ix,ix)],Zp[np.ix_(ix,ix)])
        np.fill_diagonal(M,np.nan)
        # remove class bias
        for cls in (same,~same):
            mm=cls&~np.isnan(M); M[mm]-=np.nanmean(M[mm])
        M0=np.nan_to_num(M)
        w,v=np.linalg.eigh(M0)
        out[geo]=(w[-1],w[0],v[:,-1],np.nansum(M,1)/np.sqrt(len(ix)-1))
        if verbose:
            print(h,geo,'n=%d top eig %.1f bottom %.1f (random ~ 2*sqrt(n)=%.1f)'%(len(ix),w[-1],w[0],2*np.sqrt(len(ix))))
    return ix,out
for h in sorted(set(who)):
    analyse(h,verbose=True)
ix,out=analyse('toorisugari',anon_only=True,verbose=True)
for geo in out:
    v=out[geo][2]; rs=out[geo][3]
    print(geo,'eigvec sorted:',np.round(np.sort(v),2))
    print(geo,'rowsum z: ',np.round(rs,1))
