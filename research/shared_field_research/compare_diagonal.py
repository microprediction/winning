from spectral_backbone import prepare,sample,credit
from bench_spectral import fixture
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from time import perf_counter
from coupled import prepare as diagonal

def diag_sample(mu,p,m,seed):
    A,a0,d=p;n=len(mu);sob=qmc.Sobol(2*n,scramble=True,seed=seed);total=np.zeros(n);ss=0.
    for j in range(0,2**m,4096):
        z=ndtri(sob.random(min(4096,2**m-j)));a=mu+z[:,:n]@A.T;b=mu+z[:,:n]*a0;e=d*z[:,n:];v=credit(a,(a+e).min(1),d)-credit(b,(b+e).min(1),d);total+=v.sum(0);ss+=(v*v).sum()
    total/=2**m;return total,ss/2**m-total@total

if __name__=='__main__':
    for n,case in [(32,'spike'),(64,'spike'),(32,'diffuse')]:
        mu,C=fixture(n,case)
        for name,p in [('prior diagonal',diagonal(C)),('rank1',prepare(C,1))]:
            t=perf_counter();r=[diag_sample(mu,p,15,s) if name=='prior diagonal' else sample(mu,p,np.zeros(n),15,s) for s in range(4)];print(n,case,name,'sec',(perf_counter()-t)/4,'sd',np.sqrt(np.mean(np.array([x[0] for x in r]).var(0,ddof=1))),'trace',np.mean([x[1] for x in r]),flush=True)
