import numpy as np
import pytest
from scipy.special import ndtr
from scipy.integrate import quad
from markov_tree import tree_shares, tree_covariance


def exact_three(mu, C):
    out = []
    for i in range(3):
        j = np.delete(np.arange(3), i)
        d = C[np.ix_(j,j)] - C[j,i,None] - C[i,j][None,:] + C[i,i]
        sd = np.sqrt(np.diag(d)); a = (mu[j]-mu[i])/sd
        r = d[0,1]/np.prod(sd)
        out.append(quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)*
                        ndtr((a[1]-r*z)/np.sqrt(1-r*r)), -np.inf, a[0],
                        epsabs=1e-12, epsrel=1e-12)[0])
    return np.array(out)


def test_binary():
    mu = np.array([-.4,.5]); sd = np.array([.8,1.2]); r = np.array([0,-.7]); p = [-1,0]
    C = tree_covariance(sd,p,r)
    ref = ndtr((mu[1]-mu[0])/np.sqrt(C[0,0]+C[1,1]-2*C[0,1]))
    np.testing.assert_allclose(tree_shares(mu,sd,p,r),[ref,1-ref],atol=1e-10)


@pytest.mark.parametrize('seed',range(10))
def test_three(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=3); sd = rng.uniform(.6,1.4,3); p = [-1,0,0]
    r = np.r_[0,rng.uniform(-.9,.9,2)]
    C = tree_covariance(sd,p,r)
    np.testing.assert_allclose(tree_shares(mu,sd,p,r,L=144,M=96),exact_three(mu,C),atol=1e-9)


def test_branching_and_invariance():
    rng=np.random.default_rng(41); n=18
    p=np.r_[-1,(np.arange(1,n)-1)//2]; r=np.r_[0,rng.uniform(-.85,.85,n-1)]
    mu=rng.normal(0,.6,n); sd=rng.uniform(.8,1.2,n)
    C=tree_covariance(sd,p,r)
    assert np.max(abs(np.linalg.inv(C)[(p[:,None]!=np.arange(n)) & (p[None,:]!=np.arange(n)[:,None]) & ~np.eye(n,dtype=bool)])) < 1e-12
    a=tree_shares(mu,sd,p,r,L=96,M=72)
    assert abs(a.sum()-1)<1e-9 and np.all(a>=0)
    perm=rng.permutation(n); inv=np.argsort(perm)
    pp=np.where(p[perm]>=0,inv[np.maximum(p[perm],0)],-1)
    b=tree_shares(3+2*mu[perm],2*sd[perm],pp,r[perm],L=96,M=72)
    np.testing.assert_allclose(a[perm],b,atol=1e-12)
    # Re-root at old node 1. Standardized edges have the same reverse rho.
    pp=p.copy(); rr=r.copy(); pp[0]=1; pp[1]=-1; rr[0]=rr[1]; rr[1]=0
    np.testing.assert_allclose(a,tree_shares(mu,sd,pp,rr,L=96,M=72),atol=1e-12)
    # Independently sample the original dense Gaussian.
    m=400_000; draws=mu+rng.standard_normal((m,n))@np.linalg.cholesky(C).T
    mc=np.bincount(draws.argmin(1),minlength=n)/m
    assert np.all(abs(a-mc)<5*np.sqrt(a*(1-a)/m)+1e-5)


def test_independence():
    mu=np.array([-1.,0.,1.,.5]);sd=np.array([.7,1,1.4,.9]);n=len(mu)
    p=tree_shares(mu,sd,[-1,0,0,1],np.zeros(n))
    ref=[]
    for i in range(n):
        j=np.delete(np.arange(n),i)
        ref.append(quad(lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)*np.prod(ndtr((mu[j]-mu[i]-sd[i]*x)/sd[j])), -np.inf,np.inf,epsabs=1e-12)[0])
    np.testing.assert_allclose(p,ref,atol=1e-9)


def test_refusals():
    for p,r in [([-1,2,1],[0,.2,.3]),([-1,0,1],[0,1,.3]),([-1,0,0],[.5,.2,.3]),([-1,-1,0],[0,.2,.3])]:
        with pytest.raises(ValueError): tree_shares([0,0,0],[1,1,1],p,r)
