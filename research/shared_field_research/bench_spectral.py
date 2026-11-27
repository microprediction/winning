import numpy as np
from time import perf_counter
from spectral_backbone import prepare,field,sample
from bench_coupled import ghk as raw_ghk

def ghk(mu,cov,m,seed):
    p=raw_ghk(mu,cov,m,seed)
    return p/p.sum()

def fixture(n,case):
    rng=np.random.default_rng(9011);mu=rng.normal(0,.4,n);G=rng.normal(size=(n,n));W=G@G.T/n;v=rng.normal(0,.65,(n,1));v-=v.mean(0)
    C=np.eye(n)+.12*W
    if case=='spike':C+=v@v.T
    if case=='wishart':C=W+.05*np.eye(n)
    if case=='diffuse':u=v[:,0]/np.linalg.norm(v);C=np.eye(n)-.99999*np.outer(u,u)
    return mu,C

def run():
    for n,case in [(32,'spike'),(64,'spike'),(32,'weak'),(32,'wishart'),(32,'diffuse')]:
        mu,C=fixture(n,case)
        t=perf_counter();refs=np.array([ghk(mu,C,14 if n==32 else 13,s) for s in range(100,104)]);ref=refs.mean(0)
        print(n,case,'reference sec',perf_counter()-t,'refspread',np.ptp(refs,axis=0).max(),'refsd',np.sqrt(np.mean(refs.var(0,ddof=1))),flush=True)
        def report(label,prep,ps,dt,vs=None):
            print(label,'prep',round(prep,5),'sec',round(dt/4,5),'rmse',np.sqrt(np.mean((ps-ref)**2)),'medmax',np.median(np.max(abs(ps-ref),axis=1)),'sd',np.sqrt(np.mean(ps.var(0,ddof=1))),'trace',np.mean(vs) if vs is not None else None,'min',ps.min(),'mass',abs(ps.sum(1)-1).max(),flush=True)
        for m in (10,12):
            t=perf_counter();ps=np.array([ghk(mu,C,m,s) for s in range(4)]);report(('GHK',m),0,ps,perf_counter()-t)
        for rank in (0,1):
            t=perf_counter();p=prepare(C,rank);p0=field(mu,p[3],p[4]);pre=perf_counter()-t
            finer=field(mu,p[3],p[4],order=41,points=1025)
            print('rank',rank,'backbone refinement',abs(finer-p0).max(),'p0 mass',p0.sum(),'root residual',np.linalg.norm(p[0]-p[1])**2,'p0 RMSE',np.sqrt(np.mean((p0-ref)**2)),flush=True)
            for m in (13,15):
                t=perf_counter();ans=[sample(mu,p,p0,m,s) for s in range(4)];report(('CV',rank,m),pre,np.array([a[0] for a in ans]),perf_counter()-t,[a[1] for a in ans])
if __name__=='__main__':run()
