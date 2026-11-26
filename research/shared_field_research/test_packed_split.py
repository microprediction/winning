import numpy as np
import pytest
from scipy.linalg import helmert
from scipy.special import ndtr
from packed_split import packed_split
from hazard_draw import draw_shares,contrast_split


def check(C,A,d):
    B=helmert(len(C),full=False).T
    np.testing.assert_allclose(B.T@(A@A.T+np.diag(d))@B,B.T@C@B,rtol=1e-10,atol=1e-10*np.max(abs(C)))
    assert np.all(d>0)

@pytest.mark.parametrize('n',[2,8,32])
def test_random(n):
    rng=np.random.default_rng(812+n); G=rng.normal(size=(n,n)); C=G@G.T/n+.02*np.eye(n)
    A,d=packed_split(C);check(C,A,d)
    perm=rng.permutation(n); Cp=C[np.ix_(perm,perm)];check(Cp,*packed_split(Cp))


def test_clone_and_bound():
    n=32; C=np.eye(n); C[0,1]=C[1,0]=.9995
    A,d=packed_split(C);check(C,A,d)
    assert d[0]+d[1]<=.001+1e-12
    assert np.median(d[2:])>.7
    _,ds=contrast_split(C);assert np.median(d)>100*ds[0]


def test_common_shock_and_scale():
    rng=np.random.default_rng(1);G=rng.normal(size=(10,10));C=G@G.T+np.eye(10)
    A,d=packed_split(C);A2,d2=packed_split(9*C+20*np.ones_like(C))
    np.testing.assert_allclose(d2,9*d,rtol=1e-9)
    check(9*C+20*np.ones_like(C),A2,d2)


def test_binary_exact():
    C=np.array([[1.,.4],[.4,2.]])
    mu=np.array([.3,-.2]);p=draw_shares(mu,*packed_split(C),m=16,seed=99)
    exact=ndtr((mu[1]-mu[0])/np.sqrt(C[0,0]+C[1,1]-2*C[0,1]))
    assert abs(p[0]-exact)<3e-4
    assert np.all(p>=0) and abs(p.sum()-1)<1e-14


def test_refuse_singular():
    with pytest.raises(ValueError):packed_split(np.ones((3,3)))
    with pytest.raises(ValueError):packed_split(np.diag([1.,-3.]))
