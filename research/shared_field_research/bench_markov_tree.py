"""Standalone single-thread benchmark. OPENBLAS_NUM_THREADS=1 python bench_markov_tree.py"""
import numpy as np
from scipy.special import log_ndtr, ndtri_exp, logsumexp
from scipy.stats import qmc
from time import perf_counter
from markov_tree import tree_shares, tree_covariance


def ordered_cholesky(a,C):
    # Genz-style greedy variable ordering: choose the smallest conditional
    # CDF at expected truncated draws, update the Schur factor in place.
    a=a.copy(); C=C.copy(); n=len(a); L=np.zeros_like(C); z=np.zeros(n)
    for k in range(n):
        diag=np.diag(C)[k:]-np.sum(L[k:,:k]**2,axis=1)
        b=(a[k:]-L[k:,:k]@z[:k])/np.sqrt(diag)
        j=k+np.argmin(b)
        a[[k,j]]=a[[j,k]]; C[[k,j],:]=C[[j,k],:]; C[:,[k,j]]=C[:,[j,k]]
        L[[k,j],:k]=L[[j,k],:k]
        L[k,k]=np.sqrt(C[k,k]-L[k,:k]@L[k,:k])
        L[k+1:,k]=(C[k+1:,k]-L[k+1:,:k]@L[k,:k])/L[k,k]
        bk=(a[k]-L[k,:k]@z[:k])/L[k,k]
        z[k]=-np.exp(-bk*bk/2-.5*np.log(2*np.pi)-log_ndtr(bk))
    return a,L


def ghk(mu,cov,m,seed):
    # Min-win contrasts, reordered GHK / Genz SOV, randomized Sobol;
    # includes per-alternative covariance formation and ordering.
    n=len(mu); p=np.zeros(n)
    u=qmc.Sobol(n-1,scramble=True,seed=seed).random_base2(m)
    logu=np.log(u)
    for i in range(n):
        idx=np.delete(np.arange(n),i)
        c=cov[np.ix_(idx,idx)]-cov[idx,i,None]-cov[i,idx][None,:]+cov[i,i]
        a,l=ordered_cholesky(mu[idx]-mu[i],c)
        z=np.zeros_like(u); lp=np.zeros(len(u))
        for k in range(n-1):
            b=(a[k]-z[:,:k]@l[k,:k])/l[k,k]
            f=log_ndtr(b); lp+=f
            z[:,k]=ndtri_exp(logu[:,k]+f)
        p[i]=np.exp(logsumexp(lp)-np.log(len(lp)))
    return p


def run():
    rng=np.random.default_rng(23)
    for n in [32,100]:
        for mode in ['mixed','strong']:
            parent=np.r_[-1,(np.arange(1,n)-1)//2]
            mu=rng.normal(0,.6,n);sd=rng.uniform(.8,1.2,n)
            rho=np.r_[0,rng.uniform(-.85,.85,n-1) if mode=='mixed' else rng.uniform(.85,.97,n-1)]
            cov=tree_covariance(sd,parent,rho)
            ref=tree_shares(mu,sd,parent,rho,L=160,M=112)
            check=tree_shares(mu,sd,parent,rho,L=192,M=144)
            print(n,mode,'reference change',np.max(abs(check-ref)),'sum',ref.sum(),flush=True)
            for L,M in [(48,32),(64,48),(96,72)]:
                t=perf_counter();p=tree_shares(mu,sd,parent,rho,L=L,M=M); elapsed=perf_counter()-t
                print('tree',L,M,round(elapsed,4),'max',np.max(abs(p-ref)),'sum',p.sum(),flush=True)
            for m in [8,10]:
                t=perf_counter(); pp=np.array([ghk(mu,cov,m,s) for s in range(4)]); elapsed=(perf_counter()-t)/4
                print('GHK',m,round(elapsed,4),'median max',np.median(np.max(abs(pp-ref),axis=1)),'rmse',np.sqrt(np.mean((pp-ref)**2)),flush=True)
            draws=2**20; count=np.zeros(n)
            for k in range(16):
                x=mu+rng.standard_normal((draws//16,n))@np.linalg.cholesky(cov).T
                count+=np.bincount(x.argmin(1),minlength=n)
            mc=count/draws; se=np.sqrt(ref*(1-ref)/draws)
            print('MC max absolute',np.max(abs(mc-ref)),'rms z',np.sqrt(np.mean(((mc-ref)/np.maximum(se,1/draws))**2)),flush=True)


if __name__=='__main__': run()
