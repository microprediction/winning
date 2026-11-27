import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from coupled import prepare, independent_field, sample, hard_credit

def test_reconstruct():
    rng=np.random.default_rng(8);g=rng.normal(size=(9,9));C=g@g.T+np.eye(9)
    A,a0,sd=prepare(C)
    assert np.allclose(A@A.T+sd**2*np.eye(9),C)
    assert np.allclose(a0*a0+sd*sd,np.diag(C))

def test_zero_correction():
    mu=np.array([-.2,.3,.7]);D=np.array([.7,1.2,1.4]);p,_=independent_field(mu,np.sqrt(D))
    for method in ['hard_cv','smooth_cv']:
        out,v,f=sample(mu,prepare(np.diag(D)),p,10,12,method)
        assert np.allclose(out,p,atol=1e-14) and abs(v)<1e-25 and f==0

def test_binary():
    C=np.array([[1.,-.1],[-.1,1.2]]);mu=np.array([.1,-.2]);p0,_=independent_field(mu,np.sqrt(np.diag(C)))
    exact=ndtr((mu[1]-mu[0])/np.sqrt(C[0,0]+C[1,1]-2*C[0,1]));p,_,_=sample(mu,prepare(C),p0,16,4)
    assert abs(p[0]-exact)<4e-5 and abs(p.sum()-1)<1e-12

def test_three_independent_reference():
    C=np.array([[1.,-.09,.07],[-.09,1.1,-.11],[.07,-.11,.85]]);mu=np.array([-.4,.2,.6]);p0,_=independent_field(mu,np.sqrt(np.diag(C)))
    exact=[]
    for i in range(3):
        j=np.delete(np.arange(3),i);S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        s=np.sqrt(np.diag(S));a=(mu[j]-mu[i])/s;rho=S[0,1]/np.prod(s)
        exact.append(quad(lambda t:np.exp(-t*t/2)/np.sqrt(2*np.pi)*ndtr((a[1]-rho*t)/np.sqrt(1-rho*rho)),-np.inf,a[0],epsabs=1e-12)[0])
    ps=np.array([sample(mu,prepare(C),p0,16,s)[0] for s in range(4)])
    assert np.max(abs(ps.mean(0)-exact))<4e-5
    assert ps.min()>0 and np.max(abs(ps.sum(1)-1))<1e-12

def test_shift_scale():
    C=np.array([[1.,-.2],[-.2,1.4]]);mu=np.array([-.3,.4]);p0,_=independent_field(mu,np.sqrt(np.diag(C)))
    p,_,_=sample(mu,prepare(C),p0,12,9)
    q,_,_=sample(mu*3+2,prepare(C*9),p0,12,9)
    assert np.allclose(p,q,atol=2e-14)

def test_ties_and_singular_refusal():
    assert np.array_equal(hard_credit(np.array([[0.,0.,1.],[1.,0.,0.]])),[[.5,.5,0],[0,.5,.5]])
    import pytest
    with pytest.raises(ValueError):prepare(np.ones((2,2)))

def test_singular_exact_root_and_projection():
    from coupled import hard_root_correction,simplex_projection
    A=np.array([[1.],[1.],[-1.]])
    # Identical rows, symmetric latent law: exactly .25,.25,.5.
    p=hard_root_correction(np.zeros(3),A,np.zeros(3),np.zeros_like(A),np.ones(3)/3,10,2)
    assert np.allclose(p,[.25,.25,.5],atol=1e-15)
    v=np.array([.6,.5,-.1]);q=simplex_projection(v)
    assert np.allclose(q,[.55,.45,0]) and np.linalg.norm(q-p)<=np.linalg.norm(v-p)
