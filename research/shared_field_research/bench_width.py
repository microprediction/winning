"""Seeded bounded-width precision benchmark; single-thread BLAS recommended."""
import numpy as np
from time import perf_counter
from scipy.special import ndtri
from scipy.stats import qmc
from markov_width import width_shares
from bench_markov_tree import ghk

def graph(n,width,seed=42,strong=False):
    rng=np.random.default_rng(seed); Q=np.zeros((n,n)); cliques=[tuple(range(n-width,n))]
    for i in range(n-width,n):
        for j in range(i+1,n): Q[i,j]=Q[j,i]=rng.uniform(-.65,.65) if not strong else -rng.uniform(.7,1.2)
    for i in range(n-width-1,-1,-1):
        s=cliques[rng.integers(len(cliques))]
        for j in s: Q[i,j]=Q[j,i]=rng.uniform(-.65,.65) if not strong else -rng.uniform(.7,1.2)
        cliques.extend(tuple(sorted((i,)+s[:j]+s[j+1:])) for j in range(width))
    Q+=np.diag(np.abs(Q).sum(axis=1)+(0.15 if strong else 1.0))
    return rng.normal(0,.5,n),Q

def run():
    for n,width,strong in [(32,2,False),(100,2,False),(32,2,True),(16,3,False)]:
        mu,Q=graph(n,width,strong=strong); cov=np.linalg.inv(Q)
        levels=([(48,16),(64,24),(88,36),(112,44)] if n==32 and not strong
                else [(48,16),(64,24),(80,32),(88,36)] if width==2
                else [(40,10),(48,14),(64,20),(80,24)])
        results=[]
        for L,M in levels:
            t=perf_counter(); p=width_shares(mu,Q,L=L,M=M); elapsed=perf_counter()-t
            results.append((L,M,elapsed,p)); print(n,width,strong,L,M,round(elapsed,3),p.sum(),flush=True)
        ref=results[-1][-1]
        for L,M,t,p in results[:-1]: print('err',L,M,np.max(abs(p-ref)),flush=True)
        t=perf_counter(); pp=np.array([ghk(mu,cov,10,s) for s in range(4)]); elapsed=(perf_counter()-t)/4
        print('GHK',elapsed,'median max',np.median(np.max(abs(pp-ref),axis=1)),flush=True)
        pp=[]
        for s in range(4):
            u=qmc.Sobol(n,scramble=True,seed=100+s).random_base2(17)
            x=mu+ndtri(u)@np.linalg.cholesky(cov).T
            pp.append(np.bincount(x.argmin(1),minlength=n)/len(x))
        print('MC',np.max(abs(np.mean(pp,axis=0)-ref)),flush=True)

if __name__=='__main__': run()
