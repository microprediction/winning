import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from multi_cavity import prepare,credit,shares
from bench_multi_cavity import case


def test_line_geometry():
    rng=np.random.default_rng(9); mu,C=case(12,'null3'); p=prepare(C,3); A,W,J,a,sd=p
    z=rng.normal(size=(80,A.shape[1])); x=mu+z@A.T
    v=np.eye(len(mu)); v[np.arange(len(mu))[:,None],J]=a; v-=v.mean(1)[:,None]
    np.testing.assert_allclose(W@A.T,sd[:,None]*v,atol=1e-12)
    y=credit(x,z,p); expected=np.zeros_like(y)
    for i in range(len(mu)):
        slopes=A@W[i]; intercept=x-(z@W[i])[:,None]*slopes
        diff=intercept-intercept[:,[i]]; den=slopes[i]-slopes
        hi=np.min(diff[:,den>1e-10]/den[den>1e-10],axis=1,initial=np.inf)
        lo=np.max(diff[:,den<-1e-10]/den[den<-1e-10],axis=1,initial=-np.inf)
        eq=(abs(den)<=1e-10); eq[i]=False
        expected[:,i]=np.where(np.all(diff[:,eq]>0,axis=1),np.maximum(ndtr(hi)-ndtr(lo),0),0)
    np.testing.assert_allclose(y,expected,atol=1e-11)


def test_independent_reference():
    mu=np.array([-.3,.2,.8]); s=np.array([.7,1.,1.4]); C=np.diag(s*s)
    ref=np.array([quad(lambda t: norm.pdf((t-mu[i])/s[i])/s[i]*np.prod(ndtr((mu[np.arange(3)!=i]-t)/s[np.arange(3)!=i])),-np.inf,np.inf,epsabs=1e-12)[0] for i in range(3)])
    np.testing.assert_allclose(shares(mu,prepare(C,1),16,17),ref,atol=3e-5)


def test_diffuse_modes():
    for r in [3,6]:
        mu,C=case(32,'null'+str(r)); p=prepare(C,r)
        assert np.min(p[-1])>.25
        assert np.median(p[-1])>5*np.median(prepare(C,1,'greedy')[-1])


def test_scale_gauge():
    mu,C=case(20,'wishart'); p=prepare(C,3); A,W,J,a,sd=p; z=np.random.default_rng(17).normal(size=(80,A.shape[1])); x=mu+z@A.T
    v=credit(x,z,p)
    np.testing.assert_allclose(v,credit(5*x+10,z,(5*A,W,J,a,5*sd)),atol=1e-13)
    assert np.all((v>=0)&(v<=1))


def test_singular_refusal():
    import pytest
    with pytest.raises(ValueError): prepare(np.ones((4,4)),1)
