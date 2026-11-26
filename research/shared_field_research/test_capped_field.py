import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from capped_field import capped_shares, quantile_credit

def test_binary():
    p=capped_shares([0,.4],np.empty((2,0)),[.7,1.8],L=128)
    assert abs(p[0]-ndtr(.4/np.sqrt(2.5)))<2e-6

def test_capped_reference():
    mu=np.array([-.4,.7,.2,.2]); s=np.array([.8,1.7]); A=np.empty((4,0)); d=np.r_[s*s,0.,0.]
    p=capped_shares(mu,A,d,L=128)
    ref=[quad(lambda x:norm.pdf((x-mu[i])/s[i])/s[i]*ndtr((mu[1-i]-x)/s[1-i]),-np.inf,.2)[0] for i in range(2)]
    ref+= [np.prod(ndtr((mu[:2]-.2)/s))/2]*2
    np.testing.assert_allclose(p,ref,atol=2e-6,rtol=0)
    assert abs(p.sum()-1)<1e-14

def test_one_leaf():
    p=capped_shares([-.3,.2,.2],np.empty((3,0)),[1.,0.,0.],L=4)
    np.testing.assert_allclose(p,[ndtr(.5),ndtr(-.5)/2,ndtr(-.5)/2],atol=1e-15)

def test_deterministic():
    np.testing.assert_array_equal(capped_shares([1.,0.,0.],np.empty((3,0)),[0.,0.,0.],L=2),[0.,.5,.5])

def test_invariances():
    mu=np.array([-.1,.4,.2,.2]); A=np.array([[.3,-.2],[.1,.6],[.8,.2],[.8,.2]]); d=np.array([.5,1.,0.,0.]); order=[2,0,3,1]
    p=capped_shares(mu,A,d,m=9,seed=3)
    np.testing.assert_allclose(capped_shares(mu[order],A[order],d[order],m=9,seed=3),p[order],atol=1e-13)
    np.testing.assert_allclose(capped_shares(3*mu+4,3*A+2,9*d,m=9,seed=3),p,atol=1e-13)
    assert p[2]==p[3] and abs(p.sum()-1)<1e-14 and np.all(p>=0)

def test_equal_leaves():
    np.testing.assert_allclose(quantile_credit(np.zeros((3,7)),np.ones(7),np.array([1e-8,.4,.999])),1/7,atol=1e-14)

def test_precision_factor():
    from capped_field import precision_factor
    rng=np.random.default_rng(13); B=rng.normal(size=(5,3)); d=rng.uniform(.5,1.5,5); A=np.vstack([B,np.eye(3)])
    C=A@A.T+np.diag(np.r_[d,np.zeros(3)]); Q=np.linalg.inv(C); Q=(Q+Q.T)/2
    Q[:5,:5]=np.diag(np.diag(Q[:5,:5])) # exact structural zeros in test input
    A,d=precision_factor(Q,np.arange(5)); np.testing.assert_allclose(A@A.T+np.diag(d),np.linalg.inv(Q),atol=2e-14)
    import pytest
    with pytest.raises(ValueError): precision_factor(Q,[0,5])
    A,d=precision_factor(Q,[1]); np.testing.assert_allclose(A@A.T+np.diag(d),np.linalg.inv(Q),atol=2e-14)
