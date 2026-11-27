"""Seeded single-thread benchmark. Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import numpy as np
from scipy.special import log_ndtr,ndtri_exp,logsumexp
from scipy.stats import qmc
from time import perf_counter
from coupled import prepare,independent_field,sample

def ordered_cholesky(a,C):
    a=a.copy();C=C.copy();n=len(a);L=np.zeros_like(C);z=np.zeros(n)
    for k in range(n):
        diag=np.diag(C)[k:]-np.sum(L[k:,:k]**2,axis=1);b=(a[k:]-L[k:,:k]@z[:k])/np.sqrt(diag);j=k+np.argmin(b)
        a[[k,j]]=a[[j,k]];C[[k,j],:]=C[[j,k],:];C[:,[k,j]]=C[:,[j,k]];L[[k,j],:k]=L[[j,k],:k]
        L[k,k]=np.sqrt(C[k,k]-L[k,:k]@L[k,:k]);L[k+1:,k]=(C[k+1:,k]-L[k+1:,:k]@L[k,:k])/L[k,k]
        bk=(a[k]-L[k,:k]@z[:k])/L[k,k];z[k]=-np.exp(-bk*bk/2-.5*np.log(2*np.pi)-log_ndtr(bk))
    return a,L

def ghk(mu,cov,m,seed):
    n=len(mu);p=np.zeros(n);u=qmc.Sobol(n-1,scramble=True,seed=seed).random_base2(m);logu=np.log(u)
    for i in range(n):
        idx=np.delete(np.arange(n),i);c=cov[np.ix_(idx,idx)]-cov[idx,i,None]-cov[i,idx][None,:]+cov[i,i]
        a,l=ordered_cholesky(mu[idx]-mu[i],c);z=np.zeros_like(u);lp=np.zeros(len(u))
        for k in range(n-1):
            b=(a[k]-z[:,:k]@l[k,:k])/l[k,k];f=log_ndtr(b);lp+=f;z[:,k]=ndtri_exp(logu[:,k]+f)
        p[i]=np.exp(logsumexp(lp)-np.log(len(lp)))
    return p

def run():
    for n,case in [(32,'weak'),(64,'weak'),(64,'moderate'),(32,'diffuse'),(32,'hetero'),(32,'wishart')]:
        rng=np.random.default_rng(812);mu=rng.normal(0,.4,n);G=rng.normal(size=(n,n));W=G@G.T/n;C=np.eye(n)+(.04 if case=='weak' else .25)*W
        if case=='wishart':C=W+.05*np.eye(n)
        if case=='diffuse':
            u=rng.normal(size=n);u-=u.mean();u/=np.linalg.norm(u);C=np.eye(n)-.99999*np.outer(u,u)
        if case=='hetero':
            s=np.geomspace(.04,1.5,n);C=C*s[:,None]*s[None,:]
        t=perf_counter();prep=prepare(C);p0,e=independent_field(mu,np.sqrt(np.diag(C)));pre=perf_counter()-t
        refs=np.array([ghk(mu,C,13 if n==32 else 12,s) for s in [100,101,102]]);ref=refs.mean(0)
        print(n,case,'prep',pre,'refspread',np.ptp(refs,axis=0).max(),'refsum',ref.sum(),'field err estimate',e,flush=True)
        for method,m in [('ghk',10),('hard',14),('hard_cv',14),('smooth',14),('smooth_cv',14),('smooth_cv',16)]:
            t=perf_counter();results=[];var=[];flips=[]
            for seed in range(4):
                if method=='ghk':results.append(ghk(mu,C,m,seed));continue
                p,v,f=sample(mu,prep,p0,m,seed,method);results.append(p);var.append(v);flips.append(f)
            sec=(perf_counter()-t)/4;ps=np.array(results);error=ps-ref
            print(method,m,'sec',round(sec,5),'rmse',np.sqrt(np.mean(error**2)),'max',np.median(np.max(abs(error),axis=1)),'trace',np.mean(var) if var else None,'flip',np.mean(flips) if flips else None,'mass',np.max(abs(ps.sum(1)-1)),'min',ps.min(),flush=True)

def refined():
    # Reference uses the same estimator, independent higher-budget scrambles;
    # not an independent method or exact truth. GHK also scored normalized.
    for n,scale in [(32,.04),(64,.04),(64,.25)]:
        r=np.random.default_rng(812);mu=r.normal(0,.4,n);G=r.normal(size=(n,n));C=np.eye(n)+scale*(G@G.T/n);a=prepare(C);p0,_=independent_field(mu,np.sqrt(np.diag(C)))
        refs=np.array([sample(mu,a,p0,18,i)[0] for i in range(1000,1004)]);ref=refs.mean(0)
        print(n,scale,'cv ref spread',np.ptp(refs,axis=0).max(),'p0 bias RMSE',np.sqrt(np.mean((p0-ref)**2)),flush=True)
        for method,m in [('ghk',10),('ghk',12),('smooth_cv',14),('smooth_cv',16)]:
            t=perf_counter();ps=np.array([ghk(mu,C,m,s) if method=='ghk' else sample(mu,a,p0,m,s)[0] for s in range(4)]);dt=(perf_counter()-t)/4
            for name,p in [('raw',ps),('normalized',ps/ps.sum(1)[:,None])]:
                if method!='ghk' and name=='normalized':continue
                print(method,m,name,round(dt,4),'rmse',np.sqrt(np.mean((p-ref)**2)),'medmax',np.median(np.max(abs(p-ref),axis=1)), 'scramble rmse',np.sqrt(np.mean(np.var(p,axis=0,ddof=1))),flush=True)

if __name__=='__main__':
    import sys
    (refined if '--refined' in sys.argv else run)()
