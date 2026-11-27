"""Single-thread benchmark against previous iteration's ordered QMC-GHK."""
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from markov_cycle import cycle_shares, dense_precision
from bench_markov_tree import ghk


def run():
    for n in [32,100]:
        for mode in ['mixed','strong']:
            rng=np.random.default_rng(41)
            e=rng.uniform(-.6,.6,n) if mode=='mixed' else -rng.uniform(.7,1.2,n)
            d=(1 if mode=='mixed' else .3)+abs(e)+abs(np.roll(e,1))
            mu=rng.normal(0,.5,n);C=np.linalg.inv(dense_precision(d,e))
            t=perf_counter();ref=cycle_shares(mu,d,e,L=96,M=56);print(n,mode,'ref',perf_counter()-t,ref.sum(),flush=True)
            check=cycle_shares(mu,d,e,L=112,M=64);print('ref change',np.max(abs(ref-check)),check.sum(),flush=True);ref=check
            for L,M in [(64,32),(80,40)]:
                t=perf_counter();p=cycle_shares(mu,d,e,L=L,M=M)
                print('cycle',L,M,perf_counter()-t,'max',np.max(abs(p-ref)), 'sum',p.sum(),flush=True)
            for m in [8,10]:
                t=perf_counter();pp=np.array([ghk(mu,C,m,s) for s in range(4)])
                print('GHK',m,(perf_counter()-t)/4,'median max',np.median(np.max(abs(pp-ref),axis=1)), 'RMSE',np.sqrt(np.mean((pp-ref)**2)),flush=True)
            # Independent dense simulation, four 2^18-point Sobol scrambles.
            root=np.linalg.cholesky(C);mc=[]
            for seed in range(4):
                u=qmc.Sobol(n,scramble=True,seed=100+seed);count=np.zeros(n)
                for k in range(16):
                    y=mu+ndtri(u.random(2**14))@root.T
                    count+=np.bincount(y.argmin(1),minlength=n)
                mc.append(count/2**18)
            mc=np.array(mc)
            print('dense QMC mean max err',np.max(abs(mc.mean(0)-ref)), 'mean per-entry scramble sd',mc.std(0,ddof=1).mean(),flush=True)


if __name__=='__main__':run()
