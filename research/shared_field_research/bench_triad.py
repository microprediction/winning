"""Run single-threaded: python -u bench_triad.py [n] [case] [--no-genz].

Use case 'dense' for a global positive factor plus negative near-singular triads.
"""
import sys
from time import perf_counter
import numpy as np
from scipy.linalg import block_diag
from scipy.stats._qmvnt import _qmvn
from triad_field import field, factor_field


def fixture(n,case):
    rng=np.random.default_rng(90213);mu=rng.normal(0,.5,n);blocks=[]
    for j in range(n//3):
        if case=='mixed':
            A=rng.normal(size=(3,3));C=A@A.T+.4*np.eye(3);s=np.sqrt(C.diagonal());C=C/s[:,None]/s[None,:]
        else:
            C=np.full((3,3),-.5 if case=='singular' else -.49999);np.fill_diagonal(C,1.)
        s=rng.uniform(.75,1.25,3) if case!='hetero' else np.geomspace(.12,1.6,3)
        C=C*s[:,None]*s[None,:];blocks.append((C+C.T)/2)
    return mu,blocks


def genz(mu,C,budget,seed):
    rng=np.random.default_rng(seed);n=len(mu);p=np.zeros(n)
    for i in range(n):
        j=np.delete(np.arange(n),i);S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        p[i]=_qmvn(budget,S,np.full(n-1,-np.inf),mu[j]-mu[i],rng,n_batches=8)[0]
    return p


def report(name,p,ref,sec):
    p=np.atleast_2d(p);e=p-ref;ok=ref>1e-4
    print(name,'sec',round(sec,5),'rmse',np.sqrt(np.mean(e**2)),'max',np.median(np.max(abs(e),axis=1)), 'relmax',np.median(np.max(abs(e[:,ok]/ref[ok]),axis=1)), 'sd',np.sqrt(np.mean(np.var(p,axis=0,ddof=1))) if len(p)>1 else 0., 'mass',np.max(abs(p.sum(1)-1)), 'min',p.min(),flush=True)


def run(n,case,ghk):
    mu,blocks=fixture(n,case);print(n,case,flush=True)
    if case=='dense':
        v=np.random.default_rng(90314).normal(0,.25,n);C=block_diag(*blocks)+np.outer(v,v)
        ref=factor_field(mu,blocks,v,order=31,panels=1024);ref2=factor_field(mu,blocks,v,order=25,panels=512)
        print('refinement',max(abs(ref-ref2)), 'mass',abs(ref.sum()-1),flush=True)
        for L,Q in [(128,11),(256,15)]:
            t=perf_counter();p=factor_field(mu,blocks,v,order=Q,panels=L);report(f'factor field {L}/{Q}',p,ref,perf_counter()-t)
    else:
        C=block_diag(*blocks)
        p1=field(mu,blocks,panels=4096);ref=field(mu,blocks,panels=8192)
        print('refinement',max(abs(ref-p1)),'mass',abs(ref.sum()-1),flush=True)
        for L in (64,128,256,512,1024):
            t=perf_counter();p,info=field(mu,blocks,panels=L,return_info=True);dt=perf_counter()-t
            report('field '+str(L),p,ref,dt);print(info,flush=True)
    if ghk and case!='singular':
        for budget in (2048,8192):
            t=perf_counter();p=np.array([genz(mu,C,budget,s) for s in range(4)]);dt=(perf_counter()-t)/4
            print('raw Genz mass',max(abs(p.sum(1)-1)),flush=True);report('Genz '+str(budget),p/p.sum(1)[:,None],ref,dt)


if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2],'--no-genz' not in sys.argv)
