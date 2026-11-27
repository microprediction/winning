"""Research prototype: coupled sampled-minimum correction, SPD only."""
import numpy as np
from scipy.special import log_ndtr, ndtri, logsumexp
from scipy.stats import qmc
from scipy.integrate import quad_vec

def independent_field(mu, sd, eps=1e-11):
    def f(t):
        z=(t-mu)/sd; ls=log_ndtr(-z)
        return np.exp(ls.sum()-ls-z*z/2-np.log(sd)-.5*np.log(2*np.pi))
    knots=np.unique(np.r_[mu-8*sd,mu,mu+8*sd])
    p,err=quad_vec(f,knots[0],knots[-1],points=knots[1:-1],epsabs=eps,epsrel=eps)
    for lo,hi in [(-np.inf,knots[0]),(knots[-1],np.inf)]:
        a,e=quad_vec(f,lo,hi,epsabs=eps,epsrel=eps);p+=a;err+=e
    return p,err

def hard_credit(x):
    mask=x==x.min(axis=1,keepdims=True)
    return mask/mask.sum(axis=1,keepdims=True)

def prepare(C,fraction=.5):
    """SPD target, diagonal backbone, shared diagonal noise; Procrustes coupling."""
    C=np.asarray(C,float)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or not np.all(np.isfinite(C)) or not np.allclose(C,C.T,rtol=1e-13,atol=0): raise ValueError('finite symmetric covariance required')
    if not 0<fraction<1: raise ValueError('fraction must be between 0 and 1')
    lam,U=np.linalg.eigh(C)
    if lam[0]<=0: raise ValueError('SPD only')
    d=fraction*lam[0];root=(U*np.sqrt(lam-d))@U.T;a0=np.sqrt(np.diag(C)-d)
    u,_,vt=np.linalg.svd(root*a0[None,:],full_matrices=False)
    return root@(u@vt),a0,np.sqrt(d)

def hazard_credit(m,t,sd):
    z=(t[:,None]-m)/sd;h=-.5*z*z-log_ndtr(-z)
    return np.exp(h-logsumexp(h,axis=1,keepdims=True))

def sample(mu,prepared,p0,m,seed,method='smooth_cv',batch=8192):
    A,a0,sd=prepared;n=len(mu);sob=qmc.Sobol(2*n,scramble=True,seed=seed)
    total=np.zeros(n);sumsq=0.;flips=0
    for start in range(0,2**m,batch):
        z=ndtri(sob.random(min(batch,2**m-start)))
        means=mu+z[:,:n]@A.T;means0=mu+z[:,:n]*a0
        x=means+sd*z[:,n:];x0=means0+sd*z[:,n:]
        if method=='smooth_cv':v=hazard_credit(means,x.min(1),sd)-hazard_credit(means0,x0.min(1),sd)
        elif method=='smooth':v=hazard_credit(means,x.min(1),sd)
        elif method=='hard_cv':v=hard_credit(x)-hard_credit(x0)
        elif method=='hard':v=hard_credit(x)
        else:raise ValueError(method)
        total+=v.sum(0);sumsq+=np.sum(v*v);flips+=np.count_nonzero(x.argmin(1)!=x0.argmin(1))
    total/=2**m
    return total+(p0 if method.endswith('_cv') else 0),sumsq/2**m-np.sum(total*total),flips/2**m

def hard_root_correction(mu, A, mu0, A0, p0, m, seed, batch=8192):
    """Exact supplied roots (same latent dimension, zero-pad if necessary).

    Do not approximate ties via a tolerance; identical rows and means receive
    equal credit. Signed estimator, not necessarily nonnegative at finite m.
    """
    A,A0=np.asarray(A,float),np.asarray(A0,float)
    if A.shape!=A0.shape or A.ndim!=2: raise ValueError('matching root shapes required')
    n,k=A.shape
    if k==0: return hard_credit(np.asarray(mu)[None,:])[0]
    sob=qmc.Sobol(k,scramble=True,seed=seed);total=np.zeros(n)
    for start in range(0,2**m,batch):
        z=ndtri(sob.random(min(batch,2**m-start)))
        total+=(hard_credit(mu+z@A.T)-hard_credit(mu0+z@A0.T)).sum(0)
    return p0+total/2**m

def simplex_projection(p):
    """Optional Euclidean projection; biased, consistent, and never worsens L2 error."""
    x=np.sort(p)[::-1];s=np.cumsum(x)-1;j=np.flatnonzero(x>s/np.arange(1,len(x)+1))[-1]
    return np.maximum(p-s[j]/(j+1),0)
