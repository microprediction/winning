"""OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_packed_split.py"""
import numpy as np
from time import perf_counter
from packed_split import packed_split
from hazard_draw import contrast_split,draw_shares
from bench_markov_tree import ghk

def case(n,kind):
    rng=np.random.default_rng(812); mu=rng.normal(0,.4,n); G=rng.normal(size=(n,n))
    C=.2*G@G.T/n+np.eye(n)
    if kind=='clone':
        T=np.eye(n); T[1]=T[0]; C=T@C@T.T+.0005*np.eye(n); mu[1]=mu[0]+.008
    elif kind=='hetero':
        s=np.geomspace(.04,1.5,n); C=s[:,None]*C*s[None,:]
    elif kind=='wishart': C=G@G.T/n+.05*np.eye(n)
    elif kind=='diffuse':
        v=rng.normal(size=n);v-=v.mean();v/=np.linalg.norm(v); C=np.eye(n)-.9995*np.outer(v,v)
    return mu,C

def run():
    for n,kind in [(32,'clone'),(64,'clone'),(32,'hetero'),(32,'wishart'),(32,'diffuse'),(128,'clone')]:
        mu,C=case(n,kind); spl={}
        for label,fn in [('scalar',contrast_split),('packed',packed_split)]:
            t=perf_counter(); A,d=fn(C); sec=perf_counter()-t; spl[label]=(A,d)
            print(n,kind,label,'prep',round(sec,5),'d min/median/max',np.min(d),np.median(d),np.max(d),flush=True)
        refp=np.array([ghk(mu,C,13 if n==32 else 12,s+100) for s in range(4)]); ref=refp.mean(0)
        print('ref spread',np.ptp(refp,axis=0).max(),'mass',ref.sum(),flush=True)
        for label,m in [('ghk',10),('scalar',14),('packed',14),('packed',16),('direct',16)]:
            t=perf_counter(); pp=[]
            for s in range(4):
                if label=='ghk': p=ghk(mu,C,m,s)
                else: p=draw_shares(mu,*spl['packed' if label=='direct' else label],m,s,'direct' if label=='direct' else 'hazard')
                pp.append(p)
            sec=(perf_counter()-t)/4; pp=np.array(pp)
            print(label,m,round(sec,5),'medmax',np.median(abs(pp-ref).max(1)),'rmse',np.sqrt(np.mean((pp-ref)**2)),'mass',np.max(abs(pp.sum(1)-1)),flush=True)

if __name__=='__main__':run()
