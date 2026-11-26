"""Research-only: alternative-specific sparse line preintegration.
All alternatives reuse the same Gaussian draws, score and three smallest values.
"""
import numpy as np
from scipy.linalg import helmert
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


def prepare(C, pair=True):
    """Return exact contrast root and whitened sparse cavity directions (SPD contrasts)."""
    C=np.asarray(C,float); n=len(C)
    if n<2 or C.shape!=(n,n) or not np.isfinite(C).all() or np.max(abs(C-C.T))>1e-13*np.max(abs(C)):
        raise ValueError('finite symmetric covariance with n>=2 required')
    B=helmert(n,full=False).T
    C=C-C[:,[0]]-C[[0],:]+C[0,0]
    S=B.T@C@B; lam,U=np.linalg.eigh((S+S.T)/2)
    if lam[0]<=0: raise ValueError('positive definite contrast covariance required')
    lam=lam[::-1]; H=B@U[:,::-1]
    A=H*np.sqrt(lam); F=H/np.sqrt(lam); Q=F@F.T
    i=np.arange(n)
    if pair and n>2:
        # Largest reduction in precision energy from a single companion.
        gain=Q**2/np.diag(Q)[None,:]; np.fill_diagonal(gain,-np.inf)
        j=gain.argmax(1); a=-Q[i,j]/Q[j,j]
        # avoid near-common directions at n=2 (handled by singleton above)
    else:
        j=(i+1)%n; a=np.zeros(n)
    V=F+a[:,None]*F[j]; norm=np.linalg.norm(V,axis=1)
    if np.any(norm==0): raise FloatingPointError('degenerate cavity direction')
    return A,V/norm[:,None],j,a,1/norm


def credit(x,z,W,j,a,sd):
    """Each column is its own conditional expectation, not a joint simplex draw."""
    n=x.shape[1]; i=np.arange(n); rows=np.arange(len(x))[:,None]
    # Three smallest values suffice for deleting any two entries.
    k=min(3,n); idx=np.argpartition(x,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(x,idx,axis=1); order=np.argsort(vals,axis=1)
    idx=np.take_along_axis(idx,order,axis=1); vals=np.take_along_axis(vals,order,axis=1)
    cap=np.full_like(x,np.inf)
    for t in range(k-1,-1,-1):
        valid=(idx[:,[t]]!=i)&(idx[:,[t]]!=j)
        cap=np.where(valid,vals[:,[t]],cap)
    # Along the line, r is the change in contestant i (up to common shift).
    # Conditional distribution of r given orthogonal latent residual is
    # N(-sd * z @ W_i, sd**2).
    shift=z@W.T
    hi=(cap-x)/sd+shift
    diff=x[:,j]-x; den=1-a
    pos=den>0; neg=den<0; eq=den==0
    hi[:,pos]=np.minimum(hi[:,pos],diff[:,pos]/(den[pos]*sd[pos])+shift[:,pos])
    lo=np.full_like(hi,-np.inf)
    lo[:,neg]=diff[:,neg]/(den[neg]*sd[neg])+shift[:,neg]
    # Stable positive tail interval calculation.
    p=np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))
    p=np.maximum(p,0)
    p[:,eq]=np.where(diff[:,eq]>0,p[:,eq],0)
    return p


def shares(mu,prep,m=12,seed=0,batch=4096,mode='cavity'):
    A,W,j,a,sd=prep; mu=np.asarray(mu,float); n=len(mu)
    mu=mu-mu[0]
    u=qmc.Sobol(A.shape[1],scramble=True,seed=seed).random_base2(m)
    out=np.zeros(n)
    for start in range(0,len(u),batch):
        z=ndtri(np.clip(u[start:start+batch],np.finfo(float).eps,1-np.finfo(float).eps)); x=mu+z@A.T
        if mode=='direct':
            win=x==x.min(1)[:,None]; out+=(win/win.sum(1)[:,None]).sum(0)
        else: out+=credit(x,z,W,j,a,sd).sum(0)
    return out/len(u)
