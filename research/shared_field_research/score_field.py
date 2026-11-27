"""Research prototype: all-winner likelihood remainder under independent backbone.
Not a production API; SPD on contrasts only. Inspect eigenvalues and mass defect.
"""
import numpy as np
from scipy.special import log_ndtr, ndtri, ndtri_exp, logsumexp
from scipy.integrate import quad_vec
from scipy.stats import qmc

def prepare(C):
    C=np.asarray(C,float); n=len(C)
    if n<2 or C.shape!=(n,n) or not np.isfinite(C).all() or not np.allclose(C,C.T):
        raise ValueError('finite symmetric covariance required')
    P=np.eye(n)-np.ones((n,n))/n
    C=P@C@P; C=C+np.trace(C)/(n-1)/n
    sd=np.sqrt(np.diag(C)); R=C/sd[:,None]/sd[None,:]
    ev=np.linalg.eigvalsh(R)
    if ev[0]<=0: raise ValueError('SPD on contrasts required')
    K=R-np.eye(n); J=np.eye(n)-np.linalg.solve(R,np.eye(n)); c=-.5*np.linalg.slogdet(R)[1]
    return sd,K,J,c,ev

def moments(z,K):
    h=np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(-z)); v=1+z*h-h*h
    delta=z-h; kh=h@K; k=np.diag(K)
    return .5*(np.sum(h*kh+v*k,axis=-1,keepdims=True)+2*delta*kh+k*(delta*delta-v)-np.trace(K))

def field(mu,prep,tol=1e-10):
    sd,K,*_=prep; mu=np.asarray(mu,float)
    def f(t):
        z=(t-mu)/sd; ls=log_ndtr(-z); w=np.exp(ls.sum()-ls-.5*z*z-np.log(sd*np.sqrt(2*np.pi)))
        return np.stack((w,w*moments(z,K)))
    out=np.zeros((2,len(mu)))
    bounds=np.unique(np.r_[-np.inf,(mu-8*sd).min(),mu.min(),mu.max(),(mu+8*sd).max(),np.inf])
    for l,r in zip(bounds[:-1],bounds[1:]): out+=quad_vec(f,l,r,epsabs=tol,epsrel=tol)[0]
    return out

def sample(mu,prep,p0,dp,m,seed,batch=4096):
    sd,K,J,c,ev=prep;n=len(mu); sob=qmc.Sobol(2*n,scramble=True,seed=seed)
    total=np.zeros((2,n)); ss=np.zeros(2)
    for k in range(0,2**m,batch):
        u=sob.random(min(batch,2**m-k)); t=(mu+sd*ndtri(u[:,:n])).min(1)
        z=(t[:,None]-mu)/sd; ls=log_ndtr(-z); lh=-.5*z*z-ls-np.log(sd); h=np.exp(lh-logsumexp(lh,axis=1,keepdims=True))
        y=-ndtri_exp(np.log1p(-u[:,n:])+ls); delta=z-y
        jy=y@J; ky=y@K
        lr=c+.5*np.sum(y*jy,axis=1,keepdims=True)+delta*jy+.5*delta*delta*np.diag(J)
        sc=.5*(np.sum(y*ky,axis=1,keepdims=True)-np.trace(K))+delta*ky+.5*delta*delta*np.diag(K)
        r=np.expm1(lr); v=np.stack((h*r,h*(r-sc)))
        total+=v.sum(1);ss+=np.sum(v*v,axis=(1,2))
    total/=2**m
    return np.stack((p0+total[0],p0+dp+total[1])),ss/2**m-np.sum(total*total,axis=1)
