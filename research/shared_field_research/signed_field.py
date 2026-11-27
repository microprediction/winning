"""Research prototype: all-winner Fourier field for diag(D)-vv' Gaussian races.

Inputs sd>0, a=v/sd, ||a||<=1. No covariance fitting, normalization or ridge.
"""
import numpy as np
from scipy.special import wofz, roots_legendre, ndtr, erfc


def truncated_cf(z, k):
    """Unnormalized E[exp(i*k*Y) 1{Y>z}], Y standard normal."""
    # Evaluate wofz only in upper half plane; reflect for negative z.
    zp = np.abs(z)
    kp = np.where(z >= 0, k, -k)
    tail = .5*np.exp(-.5*zp*zp+1j*kp*zp)*wofz((kp+1j*zp)/np.sqrt(2))
    return np.where(z >= 0, tail, np.exp(-.5*k*k)-tail)


def field(mu, sd, a, nt=96, nk=96, cutoff=24., pad=9., batch=16):
    """Return raw shares for covariance diag(sd**2)-(sd*a)(sd*a)'.

    Finite rectangular quadrature; refine nt,nk,cutoff,pad independently.
    Requires >=3 nonzero a in the zero observation-noise case, and
    refuses possible permanent ties. No relative tail guarantee.
    """
    mu, sd, a = (np.asarray(v, dtype=float) for v in (mu,sd,a))
    if mu.ndim != 1 or sd.shape != mu.shape or a.shape != mu.shape or not all(np.all(np.isfinite(v)) for v in (mu,sd,a)) or np.any(sd<=0) or len(mu)<2:
        raise ValueError('invalid inputs')
    aa = a@a
    if aa>1+8*np.finfo(float).eps:
        raise ValueError('non-PSD covariance')
    if aa>1: # permit only the roundoff of a unit-normalized input
        a = a/np.sqrt(aa)
    noise = max(0.,1-aa) # roundoff at a unit norm, not a ridge
    if noise<=8*np.finfo(float).eps and np.count_nonzero(a)<3:
        raise ValueError('singular sparse constraint outside supported scope')
    if nt<1 or nk<1 or cutoff<=0 or pad<=0 or batch<1:
        raise ValueError('invalid quadrature')
    # Standardize common shift and performance scale. Marginal sd <= sd.
    mu = (mu-mu.min())/sd.max()
    sd = sd/sd.max()
    lo = np.min(mu-pad*sd)
    hi = np.min(mu+pad*sd)
    # Single global Gauss rule for now; narrow sd may require refinement.
    tx,tw=roots_legendre(nt); kx,kw=roots_legendre(nk)
    t=lo+(tx+1)*(hi-lo)/2; tw=tw*(hi-lo)/2
    k=(kx+1)*cutoff/2; kw=kw*cutoff/2*np.exp(-.5*noise*k*k)*np.sqrt(2/np.pi)
    p=np.zeros(len(mu))
    ka=k[None,:,None]*a[None,None,:]
    for start in range(0,nt,batch):
        sl=slice(start,start+batch)
        z=(t[sl,None]-mu)/sd
        f=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*sd)
        z=z[:,None,:]
        c=truncated_cf(z,ka)
        # Prefix/suffix avoids division by complex numbers near zero.
        left=np.ones_like(c); right=np.ones_like(c)
        left[:,:,1:]=np.cumprod(c[:,:,:-1],axis=2)
        right[:,:,:-1]=np.cumprod(c[:,:,:0:-1],axis=2)[:,:,::-1]
        win=left*right*np.exp(1j*ka*z)*f[:,None,:]
        p+=np.einsum('t,k,tki->i',tw[sl],kw,win.real)
    return p


def mixed_field(mu, sd, a, v, nh=21, **kwargs):
    """Add one ordinary positive factor vv' using outer Gaussian quadrature.

    Conditional negative factor must itself be PSD. Common factor shifts
    are removed before integrating. Refine nh independently.
    """
    from scipy.special import roots_hermitenorm
    mu, v = np.asarray(mu, float), np.asarray(v, float)
    if v.shape != mu.shape or not np.all(np.isfinite(v)) or nh<1:
        raise ValueError('invalid positive factor')
    v = v-v.mean()
    x,w=roots_hermitenorm(nh)
    return sum(wi*field(mu+xi*v, sd, a, **kwargs) for xi,wi in zip(x,w))/np.sqrt(2*np.pi)


def tail_bounds(a, cutoff, pad=9.):
    """Absolute per-share omitted-domain bounds (not quadrature bounds).

    Assumes valid a, cutoff>0, pad>0 and the field's performance window.
    """
    a=np.asarray(a,float)
    noise=max(0.,1-a@a)
    # |F_j| <= min(1, sqrt(2/pi)/|k*a_j|) by integration by parts.
    r=np.maximum(1.,cutoff*np.abs(a)/np.sqrt(2/np.pi))
    active=r>1
    q=active.sum()-active.astype(int)
    logprod=np.log(r).sum()-np.log(r)
    algebraic=np.full(len(a),np.inf)
    ok=q>=2
    algebraic[ok]=np.sqrt(2/np.pi)*cutoff*np.exp(-logprod[ok])/(q[ok]-1)
    gaussian=erfc(cutoff*np.sqrt(noise/2))/np.sqrt(noise) if noise>0 else np.inf
    return np.minimum(algebraic,gaussian)+(len(a)+1)*ndtr(-pad)
