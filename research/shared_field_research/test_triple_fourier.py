"""Reproducible standalone tests and benchmark for the triple Fourier experiment."""
import sys
import time
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import quad
from scipy.stats._qmvnt import _qmvn
from triple_fourier import field, transform, truncated_cf


def fixture(n,kind):
    rng=np.random.default_rng(34); B=[]
    for i in range(n//3):
        if kind=='hard':
            C=np.full((3,3),-.49995); np.fill_diagonal(C,1)
        else:
            a=rng.normal(size=(3,3)); C=a@a.T+.3*np.eye(3); s=np.sqrt(C.diagonal()); C=C/np.outer(s,s)
        if kind=='hetero':
            s=rng.uniform(.5,1.5,3); C=C*np.outer(s,s)
        B.append(C)
    C=block_diag(*B); b=rng.normal(size=n); b*=np.sqrt((1-(0 if kind=='singular' else 1e-5))/(b@C@b)); m=rng.normal(0,.4,n)
    return m,B,b,C-np.outer(C@b,C@b)


def genz(m,C,budget,seed):
    p=[]; rng=np.random.default_rng(seed)
    for i in range(len(m)):
        j=np.delete(np.arange(len(m)),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        p.append(_qmvn(budget,T,m[i]-m[j],np.full(len(j),np.inf),rng,n_batches=8)[0])
    p=np.array(p); return p/p.sum()


def test_price():
    for r in [-.9,-.4,.4,.9]:
        for a,b in [(1.4,2.3),(1.4,-2.3)]:
            h,j=.2,-.6; q=np.sqrt(1-r*r)
            ref=quad(lambda y: np.exp(-y*y/2+1j*(a+r*b)*y)/np.sqrt(2*np.pi)*truncated_cf((j-r*y)/q,b*q),h,12,complex_func=True,epsabs=1e-12)[0]
            np.testing.assert_allclose(transform(h,j,a,b,r,96),ref,atol=1e-11,rtol=0)


def test_arcsine():
    B=[np.array([[1.,.4,-.2],[.4,1.,.1],[-.2,.1,1.]]),np.ones((1,1))]
    C=block_diag(*B); b=np.array([.2,-.1,.3,.25]); b*=np.sqrt(.8/(b@C@b)); C=C-np.outer(C@b,C@b)
    p=[]
    for i in range(4):
        j=np.delete(np.arange(4),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i];s=np.sqrt(np.diag(T));R=T/np.outer(s,s)
        p.append(.125+np.arcsin(R[np.triu_indices(3,1)]).sum()/(4*np.pi))
    x=field(np.zeros(4),B,b,panels=256,nk=64,cutoff=32,nz=24,batch=64,method='fast')
    np.testing.assert_allclose(x,p,atol=5e-9,rtol=0)


def test_singular_symmetry():
    B=[np.array([[1.,-.25,-.25],[-.25,1.,-.25],[-.25,-.25,1.]])]*4;b=np.ones(12)/np.sqrt(6)
    x=field(np.zeros(12),B,b,panels=128,nk=40,cutoff=20,nz=16,batch=40,method='fast')
    np.testing.assert_allclose(x,1/12,atol=2e-8,rtol=0)


def test_invariance():
    m,B,b,C=fixture(24,'hetero')
    kw=dict(panels=96,nk=28,cutoff=14,nz=10,batch=28,method='fast')
    p=field(m,B,b,**kw); idx=np.arange(24).reshape(-1,3)[::-1,::-1].ravel()
    q=field(m[idx]*3+7,[9*x[::-1,::-1] for x in B[::-1]],b[idx]/3,**kw)
    np.testing.assert_allclose(q,p[idx],atol=1e-11,rtol=0)
    assert p.min()>=0 and abs(p.sum()-1)<2e-6


if __name__=='__main__':
    n=int(sys.argv[1]); kind=sys.argv[2]; m,B,b,C=fixture(n,kind)
    ref=None
    if '--no-genz' not in sys.argv:
        t=time.perf_counter(); ref=np.array([genz(m,C,32768,i+100) for i in range(4)])
        print('reference',time.perf_counter()-t, 'sd',np.sqrt(np.mean(ref.var(axis=0,ddof=1))),flush=True); ref=ref.mean(axis=0)
        for budget in [2048,8192]:
            t=time.perf_counter(); p=np.array([genz(m,C,budget,i) for i in range(4)])
            print('Genz',budget,(time.perf_counter()-t)/4,np.sqrt(np.mean((p-ref)**2)),np.sqrt(np.mean(p.var(axis=0,ddof=1))),flush=True)
    previous=None
    for name,L,K,Q,Z,bat in [('rotation',96,14,28,10,8),('rotation',192,20,40,14,8),('price',96,14,28,16,8),('fast',96,14,28,10,28),('fast',192,20,40,14,40)]:
        t=time.perf_counter(); p,info=field(m,B,b,panels=L,nk=Q,cutoff=K,nz=Z,batch=bat,method=name,return_info=True)
        err=None if ref is None else (np.sqrt(np.mean((p-ref)**2)),np.max(abs(p-ref)),np.max(abs((p-ref)[ref>1e-4]/ref[ref>1e-4])))
        print(name,L,K,Q,Z,time.perf_counter()-t,err,p.min(),info,flush=True)
        if name=='fast':
            if previous is not None: print('fast refinement',np.max(abs(p-previous)),flush=True)
            previous=p
