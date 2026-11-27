import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from markov_width import width_shares
from bench_analytic_width import graph

@pytest.mark.parametrize('seed',range(4))
def test_three(seed):
    mu,Q=graph(3,2,seed,strength=.3); C=np.linalg.inv(Q)
    p=width_shares(mu,Q,L=96,M=40,collapse=True); r=[]
    for i in range(3):
        j=np.delete(np.arange(3),i); a=mu[j]-mu[i]
        c=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        sd=np.sqrt(np.diag(c)); rho=c[0,1]/np.prod(sd)
        r.append(quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*ndtr((a[1]/sd[1]-rho*x)/np.sqrt(1-rho*rho)),-np.inf,a[0]/sd[0],epsabs=1e-12)[0])
    np.testing.assert_allclose(p,r,atol=2e-9,rtol=0)

def test_mixed_disconnected_and_invariance():
    mu,Q=graph(9,2,strength=.3); Q=np.pad(Q,((0,1),(0,1)));Q[-1,-1]=1.3;mu=np.r_[mu,.2]
    kw=dict(L=80,M=36,collapse=True)
    p=width_shares(mu,Q,**kw)
    q=width_shares(mu,Q,L=80,M=36,collapse=False)
    np.testing.assert_allclose(p,q,atol=2e-8,rtol=0)
    np.testing.assert_allclose(p.sum(),1,atol=1e-8)
    assert np.all(p>=0)
    idx=np.random.default_rng(5).permutation(len(mu))
    np.testing.assert_allclose(width_shares(mu[idx],Q[np.ix_(idx,idx)],np.argsort(idx),**kw),p[idx],atol=2e-12,rtol=0)
    np.testing.assert_allclose(width_shares(mu*5+7,Q/25,**kw),p,atol=2e-12,rtol=0)

def test_independent_and_single():
    mu=np.array([.1,.2,.3,.4]); Q=np.diag([1.,2.,3.,4.]); sd=1/np.sqrt(np.diag(Q))
    r=[quad(lambda x:np.exp(-((x-mu[i])/sd[i])**2/2)/np.sqrt(2*np.pi)/sd[i]*np.prod(ndtr((np.delete(mu,i)-x)/np.delete(sd,i))),-np.inf,np.inf,epsabs=1e-12)[0] for i in range(4)]
    np.testing.assert_allclose(width_shares(mu,Q,L=64,M=2,collapse=True),r,atol=1e-8,rtol=0)
    np.testing.assert_allclose(width_shares([0],[[1]],L=64,M=1,collapse=True),[1],atol=1e-12)
