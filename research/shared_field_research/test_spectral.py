import numpy as np
from spectral_backbone import prepare,field,sample


def test_roots_and_invariance():
    rng=np.random.default_rng(11); n=9;G=rng.normal(size=(n,n));C=G@G.T+.2*np.eye(n);P=np.eye(n)-np.ones((n,n))/n
    a,b,d,V,s=prepare(C,2)
    np.testing.assert_allclose(a@a.T+d*d*P,P@C@P,atol=1e-12)
    np.testing.assert_allclose(b@b.T+d*d*P,V@V.T+s*P,atol=1e-12)
    for cov in (C+5*np.ones((n,n)),4*C):
        aa,bb,dd,VV,ss=prepare(cov,2);factor=4 if cov[0,0]==4*C[0,0] else 1
        np.testing.assert_allclose(aa@aa.T,factor*a@a.T,atol=1e-11)
        np.testing.assert_allclose(bb@bb.T,factor*b@b.T,atol=1e-11)
    perm=rng.permutation(n);aa,bb,dd,VV,ss=prepare(C[np.ix_(perm,perm)],2)
    np.testing.assert_allclose(bb@bb.T,(b@b.T)[np.ix_(perm,perm)],atol=1e-11)


def test_exact_factor_zero_correction():
    v=np.array([-.8,-.1,.2,.7]);C=np.eye(4)+np.outer(v,v);mu=np.array([.3,-.2,.1,.2]);prep=prepare(C,1)
    p=field(mu,prep[3],prep[4],41,1025);est,var=sample(mu,prep,p,10,19)
    np.testing.assert_allclose(est,p,atol=1e-14);assert abs(var)<1e-24
    assert abs(p.sum()-1)<1e-12
    perm=[2,0,3,1]
    q=field(3*mu[perm]+7,prep[3][perm]*3,prep[4]*9,41,1025)
    np.testing.assert_allclose(q,p[perm],atol=1e-13)


def test_independent_orthant_formula():
    C=np.array([[1.,-.3,.2],[-.3,1.2,.1],[.2,.1,.8]]);mu=np.zeros(3);exact=[]
    for i in range(3):
        j,k=np.delete(np.arange(3),i);cov=C[j,k]-C[i,j]-C[i,k]+C[i,i];sd=np.sqrt((C[j,j]+C[i,i]-2*C[i,j])*(C[k,k]+C[i,i]-2*C[i,k]));exact.append(.25+np.arcsin(cov/sd)/(2*np.pi))
    prep=prepare(C,0);p0=field(mu,prep[3],prep[4]);p=np.mean([sample(mu,prep,p0,15,s)[0] for s in range(4)],axis=0)
    np.testing.assert_allclose(p,exact,atol=5e-4);assert abs(p.sum()-1)<1e-12


def test_singular_refusal():
    import pytest
    with pytest.raises(ValueError): prepare(np.zeros((4,4)),1)
