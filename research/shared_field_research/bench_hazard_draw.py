"""Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_hazard_draw.py."""
import numpy as np
from time import perf_counter
from hazard_draw import contrast_split,draw_shares
from bench_markov_tree import ghk

def run():
    for n,case in [(32,'wishart'),(64,'wishart'),(32,'factor'),(32,'sharp'),(64,'weak'),(128,'weak')]:
        rng=np.random.default_rng(731); mu=rng.normal(0,.4,n); G=rng.normal(size=(n,n))
        if case=='wishart': C=G@G.T/n+.25*np.eye(n)
        elif case=='weak': C=.15*G@G.T/n+np.eye(n)
        else:
            V=G[:,:3]/np.sqrt(3); C=V@V.T+(0.01 if case=='sharp' else .5)*np.eye(n)
        t=perf_counter(); A,d=contrast_split(C); prep=perf_counter()-t
        refp=np.array([ghk(mu,C,13 if n==32 else 12,s+100) for s in range(3)]); ref=refp.mean(0)
        print(n,case,'cond',np.linalg.cond(C),'d',d[0],'prep',prep,'ref spread',np.ptp(refp,axis=0).max(),flush=True)
        for label,m in [('ghk',10),('hazard',14),('hazard',16),('direct',16),('pca',16),('pca',18)]:
            t=perf_counter(); pp=[]
            for s in range(4):
                if label=='ghk': p=ghk(mu,C,m,s)
                elif label=='pca':
                    V,_=contrast_split(C,0); p=draw_shares(mu,V,np.zeros(n),m,s,'direct')
                else: p=draw_shares(mu,A,d,m,s,label)
                pp.append(p)
            sec=(perf_counter()-t)/4; pp=np.array(pp)
            print(label,m,round(sec,5),np.median(abs(pp-ref).max(1)),np.sqrt(np.mean((pp-ref)**2)),np.max(abs(pp.sum(1)-1)),flush=True)

if __name__=='__main__': run()
