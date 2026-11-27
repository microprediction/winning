"""Same ordered log-GHK baseline as bench_coupled, normalized."""
import numpy as np
from time import perf_counter
from score_field import prepare,field,sample
from bench_coupled import ghk
def run():
    for n,case,scale in [(32,'weak',.04),(64,'weak',.04),(32,'moderate',.25),(32,'wishart',1),(32,'diffuse',1),(32,'hetero',.04)]:
        rng=np.random.default_rng(812);mu=rng.normal(0,.4,n);G=rng.normal(size=(n,n));W=G@G.T/n;C=np.eye(n)+scale*W
        if case=='wishart':C=W+.05*np.eye(n)
        if case=='diffuse':
            u=rng.normal(size=n);u-=u.mean();u/=np.linalg.norm(u);C=np.eye(n)-.99999*np.outer(u,u)
        if case=='hetero':
            s=np.geomspace(.1,1.5,n);C=C*s[:,None]*s[None,:]
        t=perf_counter();prep=prepare(C);p0,dp=field(mu,prep);pre=perf_counter()-t
        def g(m,s):
            p=ghk(mu,C,m,s);return p/p.sum()
        refs=np.array([g(13 if n==32 else 12,s) for s in range(100,104)]);ref=refs.mean(0)
        print(n,case,'prep',round(pre,4),'eig',prep[-1][[0,-1]],'refsd',np.sqrt(np.mean(np.var(refs,axis=0,ddof=1))),'base',np.sqrt(np.mean((p0-ref)**2)),'linear',np.sqrt(np.mean((p0+dp-ref)**2)),flush=True)
        for m in [10,12]:
            t=perf_counter();ps=np.array([g(m,s) for s in range(4)]);print('ghk',m,round((perf_counter()-t)/4,4),np.sqrt(np.mean((ps-ref)**2)),np.sqrt(np.mean(np.var(ps,axis=0,ddof=1))),flush=True)
        for m in [12,14]:
            t=perf_counter();a=[sample(mu,prep,p0,dp,m,s) for s in range(4)];dt=(perf_counter()-t)/4;ps=np.array([x[0] for x in a]);v=np.mean([x[1] for x in a],axis=0)
            for j in range(2):
                p=ps[:,j];print('LR',j,m,round(dt,4),'rmse',np.sqrt(np.mean((p-ref)**2)),'sd',np.sqrt(np.mean(np.var(p,axis=0,ddof=1))),'var',v[j],'mass',np.max(abs(p.sum(1)-1)),'min',p.min(),flush=True)
if __name__=='__main__':run()
