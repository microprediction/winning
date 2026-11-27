import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from signed_rank2 import field2, sector_bounds, choose_cutoffs


def orthants(C):
    p=[]
    for i in range(4):
        j=np.delete(np.arange(4),i)
        S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        R=S/np.sqrt(np.outer(S.diagonal(),S.diagonal()))
        p.append(.125+np.arcsin(np.clip(R[np.triu_indices(3,1)],-1,1)).sum()/(4*np.pi))
    return np.array(p)


def test_orthants():
    rng=np.random.default_rng(52);a=np.linalg.qr(rng.normal(size=(4,2)))[0]*[.75,.91]
    sd=np.array([.8,1.2,.95,1.4]);C=np.diag(sd**2)-(sd[:,None]*a)@(sd[:,None]*a).T
    p=field2(np.zeros(4),sd,a,nt=112,nr=40,sectors=48,na=3,cutoffs=24)
    print('orthant max',max(abs(p-orthants(C))))
    np.testing.assert_allclose(p,orthants(C),atol=2e-10,rtol=0)


def test_independent():
    mu=np.array([-.2,.3,.9,-.1]);sd=np.array([.7,1.,1.3,.9]);p=field2(mu,sd,np.zeros((4,2)),nt=80,nr=24,sectors=8,na=1,cutoffs=12)
    q=[]
    for i in range(4):
        j=np.delete(np.arange(4),i)
        q.append(quad(lambda t:np.exp(-t*t/2)/np.sqrt(2*np.pi)*np.prod(ndtr((mu[j]-mu[i]-sd[i]*t)/sd[j])),-np.inf,np.inf,epsabs=1e-12)[0])
    np.testing.assert_allclose(p,q,atol=1e-10,rtol=0)


def test_singular_symmetric():
    n=20;theta=2*np.pi*np.arange(n)/n;a=np.sqrt(2/n)*np.stack([np.cos(theta),np.sin(theta)],axis=1)
    k=choose_cutoffs(a,24,1e-8,128)
    p=field2(np.zeros(n),np.ones(n),a,nt=96,nr=32,sectors=24,na=2,cutoffs=k)
    print('singular symmetric max',max(abs(p-1/n)),'tail',sector_bounds(a,k).sum(0).max())
    np.testing.assert_allclose(p,np.full(n,1/n),atol=2e-9,rtol=0)
    assert np.all(p>0) and abs(p.sum()-1)<1e-8


def test_invariances():
    rng=np.random.default_rng(812);n=24;mu=rng.normal(0,.4,n);a=rng.normal(size=(n,2));a-=a.mean(0);a=np.linalg.qr(a)[0]*np.sqrt(1-1e-5);sd=rng.uniform(.8,1.2,n)
    kw=dict(nt=96,nr=32,sectors=24,na=2,cutoffs=24);p=field2(mu,sd,a,**kw)
    j=rng.permutation(n);theta=.123;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    np.testing.assert_allclose(p[j],field2(3*mu[j]+7,3*sd[j],a[j]@R,**kw),atol=1e-9,rtol=0)
    assert abs(p.sum()-1)<1e-8 and np.all(p>0)


def test_bounds_and_refusal():
    rng=np.random.default_rng(12);a=np.linalg.qr(rng.normal(size=(24,2)))[0]
    k=choose_cutoffs(a,32,1e-8,256)
    assert sector_bounds(a,k).sum(0).max()<=1e-8
    with pytest.raises(ValueError):field2(np.zeros(24),np.ones(24),2*a)
    b=np.zeros((8,2));b[:2,0]=[1/np.sqrt(2),-1/np.sqrt(2)];b[2,1]=1
    with pytest.raises(ValueError):field2(np.zeros(8),np.ones(8),b)
    with pytest.raises(ValueError):choose_cutoffs(b,16,1e-8,64)
