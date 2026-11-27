import numpy as np
import pytest
from scipy.special import ndtr
from scipy.integrate import quad
from sparse_cavity import prepare,credit,shares


def test_binary_exact():
    C=np.array([[2.,-.3],[-.3,.9]]); mu=np.array([.3,-.4]); p=ndtr((mu[1]-mu[0])/np.sqrt(3.5))
    for pair in [True,False]:
        np.testing.assert_allclose(shares(mu,prepare(C,pair),m=4),[p,1-p],atol=2e-15)


def test_exact_direction_and_conditioning():
    rng=np.random.default_rng(199); n=9; G=rng.normal(size=(n,n)); C=G@G.T+.1*np.eye(n); mu=rng.normal(size=n)
    A,W,j,a,sd=prepare(C); z=rng.normal(size=(5,n-1)); x=mu+z@A.T
    D=np.eye(n); D[np.arange(n),j]=a; D-=D.mean(1)[:,None]
    np.testing.assert_allclose(W@A.T,sd[:,None]*D,atol=2e-13)
    q=credit(x,z,W,j,a,sd)
    # Independent dense line inequalities, no shared-extrema calculation.
    for r in range(len(z)):
        for i in range(n):
            b=A@W[i]; c=x[r]-b*(z[r]@W[i]); low=-np.inf; high=np.inf
            for l in range(n):
                if l==i: continue
                delta=b[i]-b[l]; rhs=c[l]-c[i]
                if delta>1e-12: high=min(high,rhs/delta)
                elif delta < -1e-12: low=max(low,rhs/delta)
                elif rhs<0: high=-np.inf
            expect=max(0.,ndtr(high)-ndtr(low))
            assert abs(q[r,i]-expect)<2e-13


def test_diffuse_near_null():
    n=32; u=np.r_[np.ones(n//2),-np.ones(n//2)]/np.sqrt(n); C=np.eye(n)-.999999*np.outer(u,u)
    single=prepare(C,False); pair=prepare(C,True)
    assert np.max(single[-1])<.006
    assert np.min(pair[-1])>.70
    assert np.min(pair[-1]/single[-1])>100


def test_independent_reference():
    mu=np.array([-.3,.2,.7]); s=np.array([.6,1.,1.3]); C=np.diag(s*s)
    ref=[]
    for i in range(3):
        j=np.delete(np.arange(3),i)
        ref.append(quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)*np.prod(ndtr((mu[j]-mu[i]-s[i]*z)/s[j])),-10,10,epsabs=1e-12)[0])
    for pair in [True,False]:
        p=shares(mu,prepare(C,pair),m=16,seed=34)
        np.testing.assert_allclose(p,ref,atol=2e-4)
        assert np.all((p>=0)&(p<=1))


def test_credit_gauge_scale_permutation():
    rng=np.random.default_rng(72); n=10; G=rng.normal(size=(n,n)); C=G@G.T+np.eye(n); mu=rng.normal(size=n)
    A,W,j,a,sd=prepare(C); z=rng.normal(size=(100,n-1)); x=mu+z@A.T
    p=credit(x,z,W,j,a,sd)
    np.testing.assert_allclose(p,credit(7*x+100,z,W,j,a,7*sd),atol=2e-14)
    perm=rng.permutation(n); inv=np.argsort(perm)
    np.testing.assert_allclose(p[:,perm],credit(x[:,perm],z,W[perm],inv[j[perm]],a[perm],sd[perm]),atol=2e-14)
    A2,*_=prepare(C+11*np.ones_like(C)); P=np.eye(n)-np.ones((n,n))/n
    np.testing.assert_allclose(A2@A2.T,P@C@P,atol=2e-13)


def test_singular_refused():
    with pytest.raises(ValueError): prepare(np.ones((3,3)))
