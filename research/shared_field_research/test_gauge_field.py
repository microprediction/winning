import time
import numpy as np
from scipy.special import ndtr
from scipy.stats._qmvnt import _qmvn
from gauge_field import contrast_gauge,remove_common_direction,diagonal_field
from signed_field import field

def cov(sd,a): return np.diag(sd**2)-np.outer(sd*a,sd*a)
def exact4(C):
    p=[]
    for i in range(4):
        j=np.delete(np.arange(4),i); T=np.eye(4)[j]-np.eye(4)[i]; S=T@C@T.T; d=np.sqrt(np.diag(S)); R=S/d[:,None]/d[None,:]
        p.append(1/8+np.arcsin(np.clip(R[np.triu_indices(3,1)],-1,1)).sum()/(4*np.pi))
    return np.array(p)

def test_sparse_singular():
    sd=np.array([1.,.8,1.3,.6]); a=np.array([.8,.6,0,0]); C=cov(sd,a)
    p=diagonal_field(np.zeros(4),sd,a,nt=160,nk=80,cutoff=24)
    assert np.max(abs(p-exact4(C)))<1e-10, (p,exact4(C))
    assert min(p)>0

def test_heterogeneous_gauge():
    rng=np.random.default_rng(382); sd=np.exp(rng.normal(0,.5,12)); A=rng.normal(size=(12,3)); A*=.99/np.linalg.norm(A,2)
    R=np.diag(sd); Ap,N,b=contrast_gauge(R,A); P=np.eye(12)-np.ones((12,12))/12
    np.testing.assert_allclose(P@R@A,P@R@Ap,atol=1e-14)
    np.testing.assert_allclose(N,np.eye(3)-Ap.T@Ap,atol=1e-14)
    assert np.linalg.eigvalsh(N-(np.eye(3)-A.T@A)).min()>-1e-14

def test_common_only():
    sd=np.array([.6,1.,1.3,.8]); a=1/sd; a/=np.linalg.norm(a); m=np.array([0.,.2,-.1,.3])
    kw=dict(nt=128,nk=40,cutoff=12)
    np.testing.assert_allclose(diagonal_field(m,sd,a,**kw),field(m,sd,np.zeros(4),**kw),atol=1e-13)

def test_invariances():
    sd=np.array([.7,1.,1.3,.8]); a=np.array([.8,.6,0.,0.]); m=np.array([0.,.2,-.1,.3]); perm=[2,0,3,1]
    kw=dict(nt=128,nk=64,cutoff=24)
    p=diagonal_field(m,sd,a,**kw)
    np.testing.assert_allclose(diagonal_field(7+3*m[perm],3*sd[perm],-a[perm],**kw),p[perm],atol=1e-12)

def test_permanent_tie_refused():
    import pytest
    with pytest.raises(ValueError,match="possible permanent tie"):
        diagonal_field(np.zeros(4),np.ones(4),np.array([1.,-1.,0.,0.])/np.sqrt(2))

def genz(m,C,budget,seed):
    n=len(m); rng=np.random.default_rng(seed); p=[]
    for i in range(n):
        j=np.delete(np.arange(n),i); T=np.eye(n)[j]-np.eye(n)[i]; d=T@m; S=T@C@T.T
        v,*_=_qmvn(budget,S,-d,np.full(n-1,np.inf),rng,n_batches=8); p.append(v)
    p=np.array(p); return p/p.sum()

def bench(n,case):
    rng=np.random.default_rng(382); sd=rng.uniform(.65,1.35,n); m=rng.normal(0,.45,n); q=1/sd; q/=np.linalg.norm(q)
    a=np.zeros(n)
    if case=='sparse': a[:3]=[.8,.5,-.3]; a/=np.linalg.norm(a)
    elif case=='pair': a[:2]=[.8,.6]
    elif case=='common':
        v=rng.normal(size=n); v-=q*(q@v); v/=np.linalg.norm(v); a=.98*q+np.sqrt(1-.98**2)*v
    elif case=='centered':
        a[:3]=[.8,.5,-.3]; a-=q*(q@a); a/=np.linalg.norm(a)
    a*=np.sqrt(1-1e-6); C=cov(sd,a); Ap,N,b=remove_common_direction(a[:,None],q)
    ref=diagonal_field(m,sd,a,nt=384,nk=256,cutoff=128)
    ref2=diagonal_field(m,sd,a,nt=512,nk=384,cutoff=192)
    print(n,case,'noise',1-a@a,N[0,0],'refdiff',np.max(abs(ref-ref2)), 'ref min/mass',ref2.min(),ref2.sum()-1,flush=True)
    for name,fun in [('raw',field),('gauge',diagonal_field)]:
        for nk,K in [(24,12),(64,32),(128,64)]:
            t=time.perf_counter(); p=fun(m,sd,a,nt=128,nk=nk,cutoff=K); dt=time.perf_counter()-t
            print(name,nk,K,dt,np.sqrt(np.mean((p-ref2)**2)),p.min(),p.sum()-1,flush=True)
    for budget in [2048,8192]:
        t=time.perf_counter(); ps=np.array([genz(m,C,budget,i) for i in range(4)]); dt=(time.perf_counter()-t)/4
        print('genz',budget,dt,np.sqrt(np.mean((ps-ref2)**2)),np.sqrt(np.mean(ps.std(axis=0,ddof=1)**2)),flush=True)
if __name__=='__main__':
    import sys
    bench(int(sys.argv[1]),sys.argv[2])
