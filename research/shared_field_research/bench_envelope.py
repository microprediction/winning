"""Run in this directory with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import numpy as np
from time import perf_counter
from envelope import prepare,shares,scan
from bench_markov_tree import ghk

def model(n,kind,seed=53):
    rng=np.random.default_rng(seed); mu=rng.normal(0,.35,n)
    if kind=='wishart':
        X=rng.normal(size=(n,n)); X/=np.linalg.norm(X,axis=1)[:,None]
        C=.3*np.eye(n)+.7*X@X.T
    elif kind=='factor':
        V=rng.normal(size=(n,3)); C=V@V.T+.3*np.eye(n)
    else:
        v=rng.uniform(-1,1,n); C=np.outer(v,v)+.01*np.eye(n)
    return mu,C

def run_case(n,kind,refm,reps):
    mu,C=model(n,kind); t=perf_counter(); prep=prepare(mu,np.linalg.cholesky(C)); prepsec=perf_counter()-t
    t=perf_counter(); refs=np.array([ghk(mu,C,refm,s+100) for s in range(reps)]); ref=refs.mean(0)
    print(n,kind,'prep',prepsec,'cond',np.linalg.cond(C),'ref sec',perf_counter()-t,'ref spread',np.max(np.ptp(refs,axis=0)),flush=True)
    methods=[('ghk',10),('envelope',15),('direct',15),('envelope',18),('direct',18)]
    if reps==4: methods[1:1]=[('envelope',12),('direct',12)]
    for method,m in methods:
        t=perf_counter(); ps=np.array([ghk(mu,C,m,s) if method=='ghk' else shares(prep,m,s,direct=method=='direct') for s in range(reps)]); sec=(perf_counter()-t)/reps
        print(method,m,'s',round(sec,4),'rmse',np.sqrt(np.mean((ps-ref)**2)),'median max',np.median(np.max(abs(ps-ref),axis=1)),'mass',np.max(abs(ps.sum(1)-1)),flush=True)

if __name__=='__main__':
    scan(np.array([[0.,0.]]),np.array([1.,-1.])) # warm optional compiler
    for n,kind,refm,reps in [(16,'wishart',14,4),(48,'wishart',14,4),(48,'factor',14,4),(16,'near',14,3),(96,'wishart',12,3),(96,'factor',12,3)]:
        run_case(n,kind,refm,reps)
