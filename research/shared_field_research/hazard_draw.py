"""Research: sample the shared minimum field, marginalize its winner label."""
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri
from scipy.stats import qmc
from scipy.linalg import helmert


def contrast_split(C, fraction=0.999):
    """SPD on contrast subspace only. Exact law modulo common shocks."""
    C=np.asarray(C,float); n=len(C)
    if C.shape!=(n,n) or n<2 or not np.all(np.isfinite(C)) or np.max(abs(C-C.T))>1e-13*np.max(abs(C)): raise ValueError('symmetric finite covariance required')
    if not 0<=fraction<1: raise ValueError('fraction in [0,1) required')
    B=helmert(n,full=False).T
    # Difference from anchor before projecting, to remove common mode.
    C=(C+C.T)/2; C0=C-C[:,[0]]-C[[0],:]+C[0,0]
    S=B.T@C0@B; lam,U=np.linalg.eigh((S+S.T)/2)
    if lam[0]<=0: raise ValueError('positive definite contrast covariance required; supply exact root for singular laws')
    d=fraction*lam[0]
    A=(B@U[:,::-1])*np.sqrt(lam[::-1]-d)
    return A,np.full(n,d)


def draw_shares(mu,A,d,m=12,seed=0,mode='hazard',batch=4096):
    """Exact factor law. Conditional credit given sampled minimum, no root solve.
    In direct mode use exactly the same points, but retain the sampled label.
    Zero d rows are deterministic given factors; equal credit on exact ties.
    """
    mu,A,d=map(lambda x:np.asarray(x,float),(mu,A,d)); n=len(mu)
    if not n or mu.shape!=(n,) or A.ndim!=2 or A.shape[0]!=n or d.shape!=(n,) or np.any(d<0) or not all(np.isfinite(x).all() for x in (mu,A,d)): raise ValueError('invalid law')
    if mode not in ('direct','hazard'): raise ValueError('mode')
    if not isinstance(m,(int,np.integer)) or m<0 or not isinstance(batch,(int,np.integer)) or batch<1: raise ValueError('resolution')
    mu=mu-mu[0]; A=A-A[:1]; s=np.sqrt(d); scale=max(np.max(abs(mu)),np.max(abs(A),initial=0),np.max(s))
    if scale==0: return np.ones(n)/n
    if not np.isfinite(scale): raise FloatingPointError('contrast overflow')
    mu,A,s=mu/scale,A/scale,s/scale
    if np.any((d>0)&(s==0)): raise FloatingPointError('scale underflow')
    leaf=s>0; k=A.shape[1]; r=int(leaf.sum()); dim=k+r
    u=qmc.Sobol(dim,scramble=True,seed=seed).random_base2(m) if dim else np.empty((1,0))
    out=np.zeros(n)
    for start in range(0,len(u),batch):
        z=ndtri(np.clip(u[start:start+batch],np.finfo(float).eps,1-np.finfo(float).eps))
        a=mu+z[:,:k]@A.T; x=a.copy(); x[:,leaf]+=z[:,k:]*s[leaf]
        t=x.min(1); winners=x==t[:,None]
        if mode=='direct' or not r:
            out+=(winners/winners.sum(1)[:,None]).sum(0); continue
        # Deterministic atom at the cap, if reached. Continuous leaf
        # ties have probability zero in ideal arithmetic.
        cap=winners[:,~leaf].any(1)
        if np.any(cap):
            w=winners[cap][:,~leaf]; out[~leaf]+=(w/w.sum(1)[:,None]).sum(0)
        if np.any(~cap):
            v=(t[~cap,None]-a[~cap][:,leaf])/s[leaf]
            lh=-.5*v*v-.5*np.log(2*np.pi)-np.log(s[leaf])-log_ndtr(-v)
            out[leaf]+=np.exp(lh-logsumexp(lh,axis=1)[:,None]).sum(0)
    return out/len(u)
