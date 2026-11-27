"""Research-only multi-companion sparse line preintegration."""
import numpy as np
from scipy.linalg import helmert, qr
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


def prepare(C,k=2,mode='pivot'):
    C=np.asarray(C,float); n=len(C)
    if n<3 or not 1<=k<=n-2 or C.shape!=(n,n) or not np.isfinite(C).all() or np.max(abs(C-C.T))>1e-13*np.max(abs(C)):
        raise ValueError('finite symmetric covariance; 1 <= k <= n-2')
    B=helmert(n,full=False).T
    C=C-C[:,[0]]-C[[0],:]+C[0,0]
    S=B.T@C@B; lam,U=np.linalg.eigh((S+S.T)/2)
    if lam[0]<=0: raise ValueError('SPD contrasts required')
    H=B@U[:,::-1]; lam=lam[::-1]; A=H*np.sqrt(lam); F=H/np.sqrt(lam)
    J=np.empty((n,k),int); a=np.empty((n,k)); W=np.empty_like(F); sd=np.empty(n)
    for i in range(n):
        ix=np.delete(np.arange(n),i)
        if mode=='pivot':
            # Choose rows spanning the k smallest contrast eigenmodes.
            j=ix[qr(H[ix,-k:].T,pivoting=True,mode='economic')[2][:k]]
        elif mode=='greedy':
            R=F.copy(); j=[]
            for _ in range(k):
                norm=np.linalg.norm(R,axis=1); score=np.abs(R@R[i])/np.maximum(norm,np.finfo(float).tiny)
                score[[i,*j]]=-np.inf; t=score.argmax(); j.append(t)
                v=R[t]/norm[t]; R-=np.outer(R@v,v)
            j=np.array(j)
        else: raise ValueError('unknown selection')
        J[i]=j; a[i]=np.linalg.lstsq(F[j].T,-F[i],rcond=None)[0]
        v=F[i]+a[i]@F[j]; norm=np.linalg.norm(v)
        if not norm>0: raise FloatingPointError('zero direction')
        W[i]=v/norm; sd[i]=1/norm
    return A,W,J,a,sd


def credit(x,z,prep):
    _,W,J,a,sd=prep; n=x.shape[1]; i=np.arange(n); k=J.shape[1]
    idx=np.argpartition(x,k+1,axis=1)[:,:k+2]
    vals=np.take_along_axis(x,idx,axis=1)
    cap=np.full_like(x,np.inf)
    for t in range(k+2):
        v=idx[:,[t]]; valid=(v!=i)
        for l in range(k): valid &= v!=J[:,l]
        cap=np.minimum(cap,np.where(valid,vals[:,[t]],np.inf))
    shift=z@W.T; hi=(cap-x)/sd+shift; lo=np.full_like(hi,-np.inf); ok=np.ones_like(x,dtype=bool)
    for l in range(k):
        den=1-a[:,l]; diff=x[:,J[:,l]]-x; pos=den>0; neg=den<0; eq=den==0
        hi[:,pos]=np.minimum(hi[:,pos],diff[:,pos]/(den[pos]*sd[pos])+shift[:,pos])
        lo[:,neg]=np.maximum(lo[:,neg],diff[:,neg]/(den[neg]*sd[neg])+shift[:,neg])
        ok[:,eq]&=diff[:,eq]>0
    return np.where(ok,np.maximum(0,np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))),0)


def shares(mu,prep,m,seed,batch=2048):
    A=prep[0]; mu=np.asarray(mu)-mu[0]; u=qmc.Sobol(A.shape[1],scramble=True,seed=seed).random_base2(m); out=np.zeros(len(mu))
    for start in range(0,len(u),batch):
        z=ndtri(np.clip(u[start:start+batch],np.finfo(float).eps,1-np.finfo(float).eps)); x=mu+z@A.T
        out+=credit(x,z,prep).sum(0)
    return out/len(u)
