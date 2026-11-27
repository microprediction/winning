import numpy as np
from scipy.linalg import block_diag
from hybrid_pair import field


def fixture(n=4,seed=2026,eps=1e-5,hetero=False):
    rng=np.random.default_rng(seed); blocks=[]
    for j in range(n//2):
        r=(-1)**j*(.65+.25*rng.random()); s=rng.uniform(.7,1.3,2) if hetero else np.ones(2)
        blocks.append(np.array([[1,r],[r,1]])*np.outer(s,s))
    if n%2: blocks.append(np.ones((1,1)))
    C=block_diag(*blocks); b=rng.normal(size=n); b*=np.sqrt((1-eps)/(b@C@b)); mu=rng.normal(0,.4,n)
    return mu,blocks,b,C-np.outer(C@b,C@b)


def arcsine(C):
    p=[]; n=len(C)
    for i in range(n):
        j=np.delete(np.arange(n),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]; s=np.sqrt(np.diag(T)); R=T/np.outer(s,s)
        p.append(.125+np.arcsin(np.clip(R[np.triu_indices(3,1)],-1,1)).sum()/(4*np.pi))
    return np.array(p)


def test_small():
    mu,B,b,C=fixture(4,eps=.2,hetero=True); p=field(mu*0,B,b,panels=512,nk=72,cutoff=24)
    np.testing.assert_allclose(p,arcsine(C),atol=2e-8,rtol=0)


def test_singular_symmetry():
    B=[np.array([[1.,-.6],[-.6,1.]])]*8; b=np.ones(16)/np.sqrt(16*.4)
    p=field(np.zeros(16),B,b,panels=384,nk=64,cutoff=18)
    np.testing.assert_allclose(p,1/16,atol=2e-8,rtol=0)


def test_invariances():
    mu,B,b,C=fixture(12,eps=1e-5,hetero=True); kw=dict(panels=256,nk=48,cutoff=20)
    p=field(mu,B,b,**kw)
    np.testing.assert_allclose(p,field(mu*3+8,[9*x for x in B],b/3,**kw),atol=1e-12)
    idx=np.arange(12).reshape(-1,2)[::-1,::-1].ravel()
    np.testing.assert_allclose(p[idx],field(mu[idx],[x[::-1,::-1] for x in B[::-1]],b[idx],**kw),atol=1e-12)
    assert p.min()>=0 and abs(p.sum()-1)<1e-6


def test_bad():
    import pytest
    with pytest.raises(ValueError): field([0,0],[np.eye(2)],[1,1])
    with pytest.raises(ValueError): field([0,0],[np.eye(2)],[1,0])
