import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.sparse import csr_matrix
from markov_width import prepare, width_shares
from bench_width import graph

@pytest.mark.parametrize('seed',range(5))
def test_three(seed):
    mu,Q=graph(3,2,seed); C=np.linalg.inv(Q)
    p=width_shares(mu,Q,L=80,M=32); r=[]
    for i in range(3):
        j=np.delete(np.arange(3),i); a=mu[j]-mu[i]
        c=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        sd=np.sqrt(np.diag(c)); rho=c[0,1]/np.prod(sd)
        r.append(quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*ndtr((a[1]/sd[1]-rho*x)/np.sqrt(1-rho*rho)),-np.inf,a[0]/sd[0],epsabs=1e-12)[0])
    np.testing.assert_allclose(p,r,atol=2e-9,rtol=0)

@pytest.mark.parametrize('width',[1,2,3])
def test_selected_inverse(width):
    mu,Q=graph(20,width); perm=np.arange(20)[::-1]
    _,_,_,_,sd=prepare(csr_matrix(Q[np.ix_(perm,perm)]),perm,max_width=width)
    np.testing.assert_allclose(sd,np.sqrt(np.diag(np.linalg.inv(Q))),rtol=1e-12)

def test_cycle_fill():
    n=7; Q=2*np.eye(n); i=np.arange(n); j=(i+1)%n
    Q[i,j]=Q[j,i]=-.4; mu=np.zeros(n)
    p=width_shares(mu,Q,L=64,M=24,max_width=2)
    np.testing.assert_allclose(p,np.ones(n)/n,atol=1e-7,rtol=0)
    with pytest.raises(ValueError): prepare(Q,max_width=1)

def test_invariances():
    mu,Q=graph(8,2); p=width_shares(mu,Q,L=64,M=24)
    idx=np.random.default_rng(5).permutation(8)
    other=width_shares(mu[idx],Q[np.ix_(idx,idx)],np.argsort(idx),L=64,M=24)
    np.testing.assert_allclose(other,p[idx],atol=2e-12,rtol=0)
    np.testing.assert_allclose(width_shares(mu*5+7,Q/25,L=64,M=24),p,atol=2e-12,rtol=0)

def test_disconnected():
    mu=np.array([.1,.2,.3,.4]); Q=np.diag([1.,2.,3.,4.]); sd=1/np.sqrt(np.diag(Q))
    r=[quad(lambda x:np.exp(-((x-mu[i])/sd[i])**2/2)/np.sqrt(2*np.pi)/sd[i]*np.prod(ndtr((np.delete(mu,i)-x)/np.delete(sd,i))),-np.inf,np.inf,epsabs=1e-12)[0] for i in range(4)]
    np.testing.assert_allclose(width_shares(mu,Q,L=64,M=24),r,atol=1e-8,rtol=0)

def test_refuses_indefinite():
    with pytest.raises(ValueError): prepare([[1,2],[2,1]])
    with pytest.raises(ValueError): prepare([[1,1],[1,1]])
