"""Research: heterogeneous independent-noise packing in contrast space."""
import numpy as np
from scipy.linalg import helmert


def packed_split(C, barrier=.25, sweeps=12, weights=None):
    """Exact contrast covariance split A A' + diag(d), no fitted residual.

    Coordinate ascent on sum w_i log(d_i) + barrier logdet(R), where
    R=I-sum_i d_i v_i v_i' is the whitened contrast residual. Fixed
    sweeps approximate the optimization, NOT the covariance. SPD contrast
    required; singular laws should use a supplied exact root instead.
    """
    C=np.asarray(C,float); n=len(C)
    if C.shape!=(n,n) or n<2 or not np.all(np.isfinite(C)) or np.max(abs(C-C.T))>1e-13*np.max(abs(C)): raise ValueError('symmetric finite covariance required')
    if not np.isfinite(barrier) or barrier<=0 or not isinstance(sweeps,(int,np.integer)) or sweeps<1: raise ValueError('positive barrier and sweeps required')
    w=np.ones(n) if weights is None else np.asarray(weights,float)
    if w.shape!=(n,) or not np.all(np.isfinite(w)) or np.any(w<=0): raise ValueError('positive weights required')
    B=helmert(n,full=False).T
    C=(C+C.T)/2; C0=C-C[:,[0]]-C[[0],:]+C[0,0]
    S=B.T@C0@B; S=(S+S.T)/2
    scale=np.max(abs(S))
    if not np.isfinite(scale) or scale<=0: raise ValueError('positive definite contrast covariance required')
    S=S/scale
    lam,U=np.linalg.eigh(S)
    if lam[0]<=0: raise ValueError('positive definite contrast covariance required')
    V=(U.T@B.T)/np.sqrt(lam)[:,None]
    d=np.full(n,.5*lam[0]); R=np.eye(n-1)-(V*d)@V.T
    for _ in range(sweeps):
        np.linalg.cholesky(R)  # reject numerical loss of feasibility
        H=np.linalg.inv(R)
        for i in range(n):
            v=V[:,i]; u=H@v; h=v@u
            if not np.isfinite(h) or h<=0: raise FloatingPointError('invalid inverse residual')
            new=w[i]/(w[i]+barrier)*(d[i]+1/h)
            delta=new-d[i]; den=1-delta*h
            if not np.isfinite(den) or den<=0: raise FloatingPointError('invalid downdate')
            R-=delta*np.outer(v,v)
            H+=delta/den*np.outer(u,u)
            d[i]=new
        # Rebuild to avoid accumulating downdate roundoff.
        R=np.eye(n-1)-(V*d)@V.T; R=(R+R.T)/2
    # Construct the residual in ORIGINAL contrast coordinates, verify SPD.
    T=S-(B.T*d)@B; T=(T+T.T)/2
    l,Q=np.linalg.eigh(T)
    if l[0]<=0 or not np.all(np.isfinite(l)): raise FloatingPointError('nonpositive residual; no silent clipping')
    A=(B@Q[:,::-1])*np.sqrt(l[::-1])*np.sqrt(scale)
    return A,d*scale
