"""Seeded capped-field benchmarks. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from time import perf_counter
from capped_field import capped_shares,precision_factor
from bench_markov_tree import ghk

def graph(n,k,sharp=False,strength=1.):
    rng=np.random.default_rng(79); A=strength*rng.normal(size=(n-k,k))/np.sqrt(k); d=rng.uniform(.5,1.5,n-k)*(.015 if sharp else 1.)
    G=rng.normal(size=(k,k)); C=G@G.T/k+.5*np.eye(k); B=np.vstack([A,np.eye(k)])@np.linalg.cholesky(C)
    return rng.normal(0,.35,n),B,np.r_[d,np.zeros(k)]

def direct(mu,A,d,m,seed):
    n,k=A.shape; leaf=d>0; z=ndtri(qmc.Sobol(k+leaf.sum(),scramble=True,seed=seed).random_base2(m)); x=mu+z[:,:k]@A.T; x[:,leaf]+=z[:,k:]*np.sqrt(d[leaf]); return np.bincount(x.argmin(1),minlength=n)/len(x)

def measure(mu,A,d,C,reference_m,levels):
    refs=np.array([ghk(mu,C,reference_m,s+100) for s in range(3)]); ref=refs.mean(0); print('ref spread',np.ptp(refs,axis=0).max(),'mass',ref.sum(),flush=True)
    for name,m,L in levels:
        t=perf_counter(); f=(lambda s:ghk(mu,C,m,s)) if name=='ghk' else (lambda s:direct(mu,A,d,m,s)) if name=='direct' else (lambda s:capped_shares(mu,A,d,m=m,L=L,seed=s))
        pp=np.array([f(s) for s in range(3)]); sec=(perf_counter()-t)/3
        print(name,m,L,'sec',round(sec,4),'max',np.median(np.max(abs(pp-ref),axis=1)),'rmse',np.sqrt(np.mean((pp-ref)**2)),'mass defect',max(abs(pp.sum(1)-1)),flush=True)

def run():
    for n,k,sharp,strength in [(32,2,False,.2),(32,2,True,1.),(100,2,False,.2),(48,8,False,1.)]:
        mu,A,d=graph(n,k,sharp,strength); C=A@A.T+np.diag(d); print('case',n,k,sharp,strength,'cond',np.linalg.cond(C),flush=True)
        measure(mu,A,d,C,12,[('ghk',10,0),('direct',16,0),('cap',10,1),('cap',12,1),('cap',10,4)])
    rng=np.random.default_rng(87); n=48; G=rng.normal(size=(n,n))/np.sqrt(n); C=G@G.T+.25*np.eye(n); Q=np.linalg.inv(C); Q=(Q+Q.T)/2; mu=rng.normal(0,.5,n)
    t=perf_counter(); A,d=precision_factor(Q,[np.argmin(np.diag(Q))]); print('dense prep',perf_counter()-t,'cond',np.linalg.cond(C),flush=True)
    measure(mu,A,d,C,13,[('ghk',10,0),('cap',16,1),('cap',18,1),('direct',16,0),('direct',18,0)])

def sharp_refinement():
    mu,A,d=graph(32,2,True,1.); C=A@A.T+np.diag(d)
    measure(mu,A,d,C,14,[('ghk',10,0),('cap',12,1),('cap',14,1),('cap',10,8),('cap',10,16)])

if __name__=='__main__': run()
