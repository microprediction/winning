import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import pytest
from signed_field import field, mixed_field, truncated_cf


def orthants(C):
    n=len(C);p=[]
    for i in range(n):
        j=np.delete(np.arange(n),i)
        S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        R=S/np.sqrt(np.outer(S.diagonal(),S.diagonal()))
        p.append(.125+np.arcsin(np.clip(R[np.triu_indices(3,1)],-1,1)).sum()/(4*np.pi))
    return np.array(p)


def test_cf():
    for z in (-3.,-.2,0.,1.,5.):
        for k in (-7.,0.,1.,8.):
            ref=quad(lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)*np.exp(1j*k*x),z,np.inf,complex_func=True,epsabs=1e-12)[0]
            assert abs(truncated_cf(np.array(z),np.array(k))-ref)<1e-11


def test_orthants():
    sd=np.ones(4);a=np.array([1.,2.,-1.,-2.]);a/=np.linalg.norm(a)
    for eps in (.3,1e-5,0.):
        b=a*np.sqrt(1-eps);C=np.diag(sd**2)-np.outer(sd*b,sd*b)
        p=field(np.zeros(4),sd,b,nt=192,nk=512,cutoff=256)
        print(eps,np.max(abs(p-orthants(C))))
        np.testing.assert_allclose(p,orthants(C),atol=2e-7,rtol=0)
        assert abs(p.sum()-1)<1e-8 and p.min()>0


def test_invariances_and_independent():
    rng=np.random.default_rng(17);n=16;mu=rng.normal(0,.4,n);sd=rng.uniform(.7,1.3,n);a=rng.normal(size=n);a/=np.linalg.norm(a);a*=.99999
    kw=dict(nt=128,nk=64,cutoff=24);p=field(mu,sd,a,**kw);j=rng.permutation(n)
    np.testing.assert_allclose(p,field(3*mu+7,3*sd,-a,**kw),atol=1e-12,rtol=0)
    np.testing.assert_allclose(p[j],field(mu[j],sd[j],a[j],**kw),atol=1e-12,rtol=0)
    q=field(mu,sd,0*a,**kw)
    ref=[]
    for i in range(n):
        j=np.delete(np.arange(n),i)
        ref.append(quad(lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)*np.prod(ndtr((mu[j]-mu[i]-sd[i]*t)/sd[j])),-np.inf,np.inf,epsabs=1e-12)[0])
    np.testing.assert_allclose(q,ref,atol=1e-11,rtol=0)
    v=rng.normal(0,.1,n)
    np.testing.assert_allclose(mixed_field(mu,sd,a,v,nh=9,**kw),mixed_field(mu,sd,a,v+7,nh=9,**kw),atol=1e-11,rtol=0)


def test_refusal():
    with pytest.raises(ValueError):field([0,0],[1,1],[2,2])
    with pytest.raises(ValueError):field([0,0],[1,1],[1/np.sqrt(2),-1/np.sqrt(2)])


def test_mixed_orthants():
    sd=np.array([.9,1.2,1.1,.8]);a=np.array([1.,2.,-1.,-2.])*np.sqrt(.7/10)
    v=np.array([.1,.3,-.2,.05]);C=np.diag(sd**2)-np.outer(sd*a,sd*a)+np.outer(v,v)
    p=mixed_field(np.zeros(4),sd,a,v,nh=15,nt=96,nk=48,cutoff=16)
    np.testing.assert_allclose(p,orthants(C),atol=1e-10,rtol=0)
