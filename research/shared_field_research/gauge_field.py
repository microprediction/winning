"""Contrast gauge for signed Gaussian shared fields (research only)."""
import numpy as np

def contrast_gauge(root, loadings):
    """R maps independent standard normals to performances; A maps to observations.

    Remove the observation loading along the common-shift direction
    q with R q = 1 for an invertible square root.
    Returns (A_new, noise_new, removed_row), assumes I-A.T@A is PSD.
    This routine is for explicitly supplied full-row-rank R, or roots with
    an exactly represented common-shift direction. Numerical rank decisions
    are not made here: supply q explicitly to remove_common_direction instead.
    """
    R=np.asarray(root,float); A=np.asarray(loadings,float)
    if R.ndim!=2 or A.ndim!=2 or R.shape[1]!=A.shape[0]: raise ValueError('shapes')
    # Restrict automatic calculation to square nonsingular roots.
    if R.shape[0]!=R.shape[1]: raise ValueError('supply explicit common direction for rectangular roots')
    q=np.linalg.solve(R,np.ones(R.shape[0]))
    return remove_common_direction(A,q)

def remove_common_direction(A,q):
    """q must be a nonzero latent direction mapping to a common performance shift.

    In the diagonal case use q=1/sd. No covariance fitting or ridge.
    """
    A=np.asarray(A,float); q=np.asarray(q,float)
    if A.ndim!=2 or q.shape!=(A.shape[0],) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(q)) or not np.any(q): raise ValueError('invalid inputs')
    gram=A.T@A
    N=np.eye(A.shape[1])-gram
    if np.linalg.eigvalsh(N).min() < -8*np.finfo(float).eps*max(1,A.shape[1]): raise ValueError('non-PSD signed update')
    q=q/np.max(np.abs(q)); q=q/np.linalg.norm(q)
    b=q@A
    Ap=A-np.outer(q,b)
    return Ap,N+np.outer(b,b),b

def diagonal_field(mu,sd,a,**kwargs):
    from signed_field import field
    sd=np.asarray(sd,float); a=np.asarray(a,float)
    if sd.ndim!=1 or a.shape!=sd.shape or np.any(sd<=0) or not np.all(np.isfinite(sd)): raise ValueError('invalid sd')
    # min(sd)/sd avoids overflow in reciprocal.
    Ap,_,b=remove_common_direction(a[:,None],sd.min()/sd)
    # A near-unit two-sparse contrast may be a permanent tie. Refuse
    # instead of letting roundoff in the gauge create fictitious noise.
    if a@a>=1-8*np.finfo(float).eps and np.count_nonzero(a)<=2 and abs(b[0])<=8*np.finfo(float).eps:
        raise ValueError("possible permanent tie: sparse centered singular constraint")
    return field(mu,sd,Ap[:,0],**kwargs)
