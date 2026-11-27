"""Reproducible rank-two negative factor benchmarks, one thread.
The refined Fourier field is NOT an independent large-n certificate.
Usage: python -u bench_rank2.py [diffuse|hetero|sparse] [32|64]
"""
import numpy as np
from time import perf_counter
from scipy.stats._qmvnt import _qmvn
from signed_rank2 import field2, sector_bounds, choose_cutoffs


def fixture(n,case):
    rng=np.random.default_rng(812);mu=rng.normal(0,.4,n);a=rng.normal(size=(n,2));a-=a.mean(0)
    if case=='sparse':a[8:]=0;a[:8]-=a[:8].mean(0)
    a=np.linalg.qr(a)[0]*np.sqrt(1-1e-5)
    sd=np.geomspace(.12,1.6,n) if case=='hetero' else np.ones(n)
    C=np.diag(sd**2)-(sd[:,None]*a)@(sd[:,None]*a).T
    return mu,sd,a,C


def genz(mu,C,budget,seed):
    rng=np.random.default_rng(seed);n=len(mu);p=np.zeros(n)
    for i in range(n):
        j=np.delete(np.arange(n),i)
        S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        p[i]=_qmvn(budget,S,np.full(n-1,-np.inf),mu[j]-mu[i],rng,n_batches=8)[0]
    return p


def report(name,p,ref,seconds):
    p=np.atleast_2d(p);e=p-ref;ok=ref>1e-4
    print(name,'sec',round(seconds,5),'rmse',np.sqrt(np.mean(e**2)), 'max',np.median(np.max(abs(e),axis=1)), 'relmax',np.median(np.max(abs(e[:,ok]/ref[ok]),axis=1)), 'sd',np.sqrt(np.mean(np.var(p,axis=0,ddof=1))) if len(p)>1 else 0.,'mass',np.max(abs(p.sum(1)-1)),'min',p.min(),flush=True)


def run():
    import sys
    cases=[(32,'diffuse'),(64,'diffuse'),(32,'hetero'),(32,'sparse')]
    for n,case in cases:
        if len(sys.argv)>1 and case!=sys.argv[1]:continue
        if len(sys.argv)>2 and n!=int(sys.argv[2]):continue
        mu,sd,a,C=fixture(n,case)
        print(n,case,flush=True)
        def price(nt,nr,m,na,k):return field2(mu,sd,a,nt=nt,nr=nr,sectors=m,na=na,cutoffs=k)
        # For sparse fixture even this refinement need not be accurate.
        ref=price(144 if case=='diffuse' else 160,40,28,2,32);ref2=price(128,32,24,2,24)
        print('ref delta',max(abs(ref-ref2)),'ref mass',abs(ref.sum()-1),'ref min',ref.min(),flush=True)
        settings=[(64,16,12,1,10),(80,20,16,1,12),(96,24,24,1,16)] if case=='diffuse' else [(64,16,12,1,10),(96,24,24,1,16),(144,32,24,1,16)]
        for nt,nr,m,na,k in settings:
            t=perf_counter();p=price(nt,nr,m,na,k);dt=perf_counter()-t
            report(f'field {nt}/{nr}/{m}/{na}/{k}',p,ref,dt)
            print('tailbound',sector_bounds(a,np.full(m,k)).sum(0).max(),flush=True)
        try:
            t=perf_counter();k=choose_cutoffs(a,24,1e-8,64);p=price(96,32,24,2,k);dt=perf_counter()-t
            report('adaptive',p,ref,dt);print('cutoffs',k,'tailbound',sector_bounds(a,k).sum(0).max(),flush=True)
        except ValueError as e:print('adaptive refused',e,flush=True)
        for budget in (2048,8192) if case=='diffuse' else (2048,):
            t=perf_counter();p=np.array([genz(mu,C,budget,s) for s in range(4)]);dt=(perf_counter()-t)/4
            print('raw Genz mass',np.max(abs(p.sum(1)-1)),flush=True)
            report(f'Genz {budget}',p/p.sum(1)[:,None],ref,dt)


if __name__=='__main__':run()
