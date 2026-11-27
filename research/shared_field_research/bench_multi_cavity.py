"""OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_multi_cavity.py"""
import numpy as np
from time import perf_counter
from multi_cavity import prepare, shares
from bench_markov_tree import ghk

def case(n,kind):
    rng=np.random.default_rng(812); mu=rng.normal(0,.4,n); G=rng.normal(size=(n,n)); C=G@G.T/n+.05*np.eye(n)
    if kind.startswith('null'):
        r=int(kind[4:]); V=G[:,:r]-G[:,:r].mean(0); U=np.linalg.qr(V)[0]; C=np.eye(n)-(1-1e-5)*U@U.T
    elif kind=='hetero':
        C=np.eye(n)+.2*G@G.T/n; s=np.geomspace(.04,1.5,n); C=s[:,None]*C*s[None,:]
    return mu,C

if __name__=='__main__':
    for n,kind in [(32,'null3'),(64,'null3'),(32,'null6'),(32,'wishart'),(32,'hetero')]:
        mu,C=case(n,kind); refp=np.array([ghk(mu,C,13 if n==32 else 12,s) for s in [100,101,102]]); ref=refp.mean(0)
        print(n,kind,'refspread',np.ptp(refp,axis=0).max(),flush=True)
        for k,mode,m in [(0,'ghk',10),(1,'greedy',14),(3,'pivot',14),(3,'greedy',14),(6,'pivot',14),(3,'pivot',16)]:
            t=perf_counter(); prep=prepare(C,k,mode) if k else None; pt=perf_counter()-t
            t=perf_counter(); pp=np.array([shares(mu,prep,m,s) if k else ghk(mu,C,m,s) for s in range(4)]); sec=(perf_counter()-t)/4
            print(k,mode,m,'prep',round(pt,4),'sd',round(np.median(prep[-1]),4) if k else '-', 'sec',round(sec,4),'rmse',np.sqrt(np.mean((pp-ref)**2)),'max',np.median(abs(pp-ref).max(1)),'mass',np.max(abs(pp.sum(1)-1)),flush=True)
