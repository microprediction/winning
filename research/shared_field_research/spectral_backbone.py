"""Contrast-space spectral backbone for coupled shared hazard correction.
Standalone experiment; field() reproduces the shared survival integral with
per-factor-node trapezoidal grids; no production dispatch.
"""
import numpy as np
from scipy.linalg import helmert
from scipy.special import log_ndtr, ndtri, logsumexp, roots_hermitenorm
from scipy.stats import qmc


def prepare(C, rank, fraction=.8):
    """Target SPD on contrasts; same diagonal d for both races.
    Optimal residual-root Frobenius spectral approximation with isotropic
    floor and rank positive spikes, for fixed d. No target ridge.
    """
    C=np.asarray(C,float); n=len(C)
    if C.shape!=(n,n) or not np.isfinite(C).all() or not np.allclose(C,C.T,rtol=1e-13,atol=0):
        raise ValueError('finite symmetric covariance required')
    if not 0<=rank<n-1 or not 0<fraction<1: raise ValueError('rank/fraction')
    B=helmert(n,full=False).T
    Cc=C-C.mean(0)[None,:]-C.mean(1)[:,None]+C.mean()
    lam,U=np.linalg.eigh(B.T@Cc@B); lam=lam[::-1];U=(B@U[:,::-1])
    if lam[-1]<=0: raise ValueError('positive definite on contrasts required')
    d=fraction*lam[-1];r=np.sqrt(lam-d); b=r[rank:].mean();s2=b*b+d
    r0=r.copy();r0[rank:]=b
    V=U[:,:rank]*np.sqrt(np.maximum(lam[:rank]-s2,0))
    return U*r, U*r0, np.sqrt(d),V,s2


def field(mu,V,s2,order=21,points=513):
    """Same shared survival integral as factor/core.py, per-node windows.
    Tensor GH over rank <=2 in the experiment. No renormalization.
    """
    mu=np.asarray(mu);n=len(mu);r=V.shape[1];sd=np.sqrt(s2)
    nodes,w=roots_hermitenorm(order);w=w/np.sqrt(2*np.pi)
    if r:
        ix=np.indices((order,)*r).reshape(r,-1).T
        means=mu+nodes[ix]@V.T; weights=np.prod(w[ix],axis=1)
    else:means=mu[None,:];weights=np.ones(1)
    p=np.zeros(n);u=np.linspace(0,1,points);tw=np.ones(points);tw[[0,-1]]=.5
    for a in range(0,len(means),max(1,2**20//(points*n))):
        M=means[a:a+max(1,2**20//(points*n))];W=weights[a:a+len(M)]
        lo=M.min(1)-10*sd; hi=M.max(1)+10*sd
        t=lo[:,None]+(hi-lo)[:,None]*u;z=(t[:,:,None]-M[:,None,:])/sd
        ls=log_ndtr(-z); lp=ls.sum(2,keepdims=True)-ls-.5*z*z-np.log(sd*np.sqrt(2*np.pi))
        p+=(W*(hi-lo)/(points-1))@np.einsum('mln,l->mn',np.exp(lp),tw)
    return p


def credit(m,t,sd):
    z=(t[:,None]-m)/sd;h=-.5*z*z-log_ndtr(-z)
    return np.exp(h-logsumexp(h,axis=1,keepdims=True))


def sample(mu,prep,p0,m,seed,batch=4096):
    A,A0,sd,_,_=prep;n=len(mu);sob=qmc.Sobol(2*n-1,scramble=True,seed=seed)
    total=np.zeros(n);ss=0.
    for j in range(0,2**m,batch):
        z=ndtri(sob.random(min(batch,2**m-j)));y=z[:,:n-1];e=sd*z[:,n-1:]
        a=mu+y@A.T;a0=mu+y@A0.T
        v=credit(a,(a+e).min(1),sd)-credit(a0,(a0+e).min(1),sd)
        total+=v.sum(0);ss+=np.sum(v*v)
    total/=2**m
    return p0+total,ss/2**m-total@total
