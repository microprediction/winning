import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from markov_cycle import cycle_marginals, cycle_shares, dense_precision


def case(n, seed=41):
    r=np.random.default_rng(seed)
    e=r.uniform(-.6,.6,n); d=1+abs(e)+abs(np.roll(e,1))
    return r.normal(0,.5,n),d,e


def orthant3(mu,C):
    out=[]
    for i in range(3):
        j=np.delete(np.arange(3),i)
        c=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        sd=np.sqrt(c.diagonal()); a,b=(mu[j]-mu[i])/sd
        r=c[0,1]/np.prod(sd)
        out.append(quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)*ndtr((b-r*z)/np.sqrt(1-r*r)),
                        -np.inf,a,epsabs=1e-12)[0])
    return np.array(out)


@pytest.mark.parametrize('n',[3,4,12,100])
def test_marginals(n):
    _,d,e=case(n);Q=dense_precision(d,e)
    sd,ld=cycle_marginals(d,e)
    np.testing.assert_allclose(sd**2,np.linalg.inv(Q).diagonal(),rtol=1e-12)
    assert abs(ld-np.linalg.slogdet(Q)[1])<1e-10


@pytest.mark.parametrize('seed',range(6))
def test_triangle(seed):
    mu,d,e=case(3,seed);C=np.linalg.inv(dense_precision(d,e))
    p=cycle_shares(mu,d,e,L=80,M=48)
    np.testing.assert_allclose(p,orthant3(mu,C),atol=2e-9,rtol=0)


def test_frustrated_triangle():
    mu=np.array([-.3,.15,.4]);d=np.ones(3);e=np.full(3,.85)
    C=np.linalg.inv(dense_precision(d,e))
    p=cycle_shares(mu,d,e,L=128,M=112)
    np.testing.assert_allclose(p,orthant3(mu,C),atol=2e-8,rtol=0)


def test_invariance():
    mu,d,e=case(9);kw=dict(L=88,M=56)
    p=cycle_shares(mu,d,e,**kw)
    np.testing.assert_allclose(cycle_shares(np.roll(mu,3),np.roll(d,3),np.roll(e,3),**kw),np.roll(p,3),atol=2e-12)
    np.testing.assert_allclose(cycle_shares(mu[::-1],d[::-1],np.roll(e[::-1],-1),**kw),p[::-1],atol=2e-12)
    np.testing.assert_allclose(cycle_shares(3*mu+15,d/9,e/9,**kw),p,atol=2e-12)
    assert abs(p.sum()-1)<1e-9


def test_exchangeable_and_zero_edge():
    p=cycle_shares(np.zeros(8),np.ones(8)*2,np.ones(8)*-.4,L=96,M=56)
    np.testing.assert_allclose(p,1/8,atol=1e-9)
    p=cycle_shares(np.zeros(8),np.ones(8),np.zeros(8),L=96,M=56)
    np.testing.assert_allclose(p,1/8,atol=1e-9)


def test_refuse_indefinite():
    with pytest.raises(ValueError): cycle_shares(np.zeros(3),np.ones(3),np.ones(3)*2)
