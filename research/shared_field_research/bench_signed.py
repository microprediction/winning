"""Single-thread benchmark; SciPy 1.17 private compiled Genz lattice API.
Reference is refinement of the same Fourier field, NOT an independent certificate.
"""
import numpy as np
from scipy.stats._qmvnt import _qmvn
from signed_field import field,mixed_field
from time import perf_counter


def genz(mu,cov,budget,seed):
    rng=np.random.default_rng(seed);n=len(mu);p=np.empty(n)
    for i in range(n):
        idx=np.delete(np.arange(n),i)
        C=cov[np.ix_(idx,idx)]-cov[idx,i,None]-cov[i,idx][None,:]+cov[i,i]
        p[i]=_qmvn(budget,C,np.full(n-1,-np.inf),mu[idx]-mu[i],rng,n_batches=8)[0]
    return p


def fixture(n,case):
    r=np.random.default_rng(812);mu=r.normal(0,.4,n);a=r.normal(size=n);a-=a.mean();sd=np.ones(n);v=np.zeros(n)
    if case=='hetero':sd=np.geomspace(.12,1.6,n)
    if case=='sparse':a[4:]=0;a[:4]-=a[:4].mean()
    if case=='mixed':v=r.normal(0,.3,n)
    a/=np.linalg.norm(a);a*=np.sqrt(1-1e-5)
    C=np.diag(sd**2)-np.outer(sd*a,sd*a)+np.outer(v,v)
    return mu,sd,a,v,C


def run():
    import sys
    cases=[(32,'diffuse'),(64,'diffuse'),(32,'hetero'),(32,'sparse'),(32,'mixed')]
    if "--large" in sys.argv: cases=[(128,'diffuse')]
    for n,case in cases:
        mu,sd,a,v,C=fixture(n,case)
        def price(nt,nk,k,nh):
            kw=dict(nt=nt,nk=nk,cutoff=k)
            return mixed_field(mu,sd,a,v,nh=nh,**kw) if case=='mixed' else field(mu,sd,a,**kw)
        ref=price(192,80,48,31);ref2=price(256,120,96,41)
        print(n,case,'ref_delta',np.max(abs(ref-ref2)),'ref_min',ref2.min(),flush=True)
        for nt,nk,k,nh in [(96,24,12,15),(128,48,24,21)]:
            t=perf_counter();p=price(nt,nk,k,nh);dt=perf_counter()-t
            report('field',f'{nt}/{nk}/{k}/{nh}',p[None,:],ref2,dt)
        for budget in (2048,8192):
            t=perf_counter();ps=np.array([genz(mu,C,budget,seed) for seed in range(4)]);dt=(perf_counter()-t)/4
            report('Genz',budget,ps/ps.sum(axis=1)[:,None],ref2,dt)


def report(name,budget,ps,ref,dt):
    err=ps-ref;keep=ref>1e-4
    print(name,budget,'sec',round(dt,5),'rmse',np.sqrt(np.mean(err**2)),
          'max',np.median(np.max(abs(err),axis=1)), 'relmax',np.median(np.max(abs(err[:,keep]/ref[keep]),axis=1)),
          'sd',np.sqrt(np.mean(np.var(ps,axis=0,ddof=1))) if len(ps)>1 else 0.,
          'mass',np.max(abs(ps.sum(1)-1)),'min',ps.min(),flush=True)


if __name__=='__main__':run()
