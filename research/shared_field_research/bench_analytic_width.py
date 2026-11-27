"""Seeded analytic leaf ablation; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
from time import perf_counter
import numpy as np
from markov_width import width_shares
from bench_markov_tree import ghk


def graph(n,k,seed=79,sharp=False,strength=1.):
    """Independent leaf block given a k-dimensional clique of contestants."""
    rng=np.random.default_rng(seed)
    A=strength*rng.normal(size=(n-k,k))/np.sqrt(k)
    d=rng.uniform(.5,1.5,n-k)*(0.015 if sharp else 1.)
    G=rng.normal(size=(k,k)); C=G@G.T/k+.5*np.eye(k)
    Q=np.zeros((n,n)); Q[:n-k,:n-k]=np.diag(1/d)
    Q[:n-k,n-k:]=-A/d[:,None]; Q[n-k:,:n-k]=Q[:n-k,n-k:].T
    Q[n-k:,n-k:]=np.linalg.inv(C)+A.T@(A/d[:,None]); Q=(Q+Q.T)/2
    return rng.normal(0,.35,n),Q


def run():
    cases=[
        (100,2,False,.2,[(64,28),(88,36)],[(48,12),(64,20)]),
        (64,3,False,.2,[(64,24),(80,32)],[(48,12),(64,20)]),
        (32,3,False,1.,[(80,32),(112,44)],[(48,16),(64,24)]),
        (32,2,False,1.,[(96,56),(128,80)],[(48,16),(64,24)]),
        (32,2,True,1.,[(96,56),(128,80)],[(48,16),(64,24)]),
        (16,3,False,1.,[(96,56)],[(48,16),(64,24)]),
    ]
    for n,k,sharp,strength,levels,ablation in cases:
        mu,Q=graph(n,k,sharp=sharp,strength=strength); C=np.linalg.inv(Q)
        print('case',n,k,sharp,strength,'cond',np.linalg.cond(Q),flush=True)
        refs=[]
        for L,M in levels:
            t=perf_counter()
            p=width_shares(mu,Q,L=L,M=M,max_width=k,batch=2,collapse=True)
            refs.append(p)
            print('ref',L,M,perf_counter()-t,'mass',p.sum(),flush=True)
        ref=refs[-1]
        if len(refs)>1: print('ref change',max(abs(ref-refs[0])),flush=True)
        for L,M in ablation:
            for collapse in [True,False]:
                t=perf_counter()
                p=width_shares(mu,Q,L=L,M=M,max_width=k,batch=2,collapse=collapse)
                print('collapse',collapse,L,M,perf_counter()-t,'err',max(abs(p-ref)),'mass',p.sum(),flush=True)
        t=perf_counter(); pp=np.array([ghk(mu,C,10,s) for s in range(4)])
        print('ghk',(perf_counter()-t)/4,'max',np.median(np.max(abs(pp-ref),axis=1)),flush=True)


if __name__=='__main__': run()
