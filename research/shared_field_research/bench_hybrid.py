import sys,time
import numpy as np
from scipy.stats._qmvnt import _qmvn
from hybrid_pair import field
from test_hybrid import fixture


def genz(mu,C,budget,seed):
    n=len(mu); p=[]; rng=np.random.default_rng(seed)
    for i in range(n):
        j=np.delete(np.arange(n),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        v,*_=_qmvn(budget,T,mu[i]-mu[j],np.full(n-1,np.inf),rng,n_batches=8)
        p.append(v)
    p=np.array(p); return p/p.sum()


def stats(p,ref):
    p=np.atleast_2d(p); e=p-ref; mask=ref>1e-4
    return (np.sqrt(np.mean(e**2)),np.median(np.max(abs(e),axis=1)),np.median(np.max(abs(e[:,mask]/ref[mask]),axis=1)),np.sqrt(np.mean(np.var(p,axis=0,ddof=1))) if len(p)>1 else 0)


if __name__=='__main__':
    n=int(sys.argv[1]); kind=sys.argv[2]; mu,B,b,C=fixture(n,eps=0 if kind=='singular' else 1e-5,hetero=kind=='hetero')
    if kind=='hard':
        B=[np.array([[1.,r],[r,1.]]) for r in np.resize([-.99999,.99999],n//2)]; from scipy.linalg import block_diag
        C=block_diag(*B); b*=np.sqrt((1-1e-5)/(b@C@b)); C=C-np.outer(C@b,C@b)
    def run(L,Q,K):
        t=time.perf_counter(); p,info=field(mu,B,b,panels=L,nk=Q,cutoff=K,return_info=True); return p,time.perf_counter()-t,info
    ref,t,info=run(1024,96,32); finer,tf,inf=run(1536,112,40)
    print('ref',t,tf,'maxdiff',np.max(abs(ref-finer)), 'info',inf,'min',finer.min(),flush=True)
    for L,Q,K in [(96,24,10),(192,32,14),(384,48,20)]:
        p,t,info=run(L,Q,K); print('field',L,Q,K,t,stats(p,finer),info,'min',p.min(),flush=True)
    if kind!='singular':
        for budget in [2048,8192] if n<=32 else [2048]:
            t=time.perf_counter(); p=np.array([genz(mu,C,budget,i) for i in range(4)]); print('genz',budget,(time.perf_counter()-t)/4,stats(p,finer),flush=True)
