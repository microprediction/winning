import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.linalg import block_diag
from triad_field import bvn, field


def bvn_reference(h,k,r):
    if abs(r)==1:
        return ndtr(min(h,k)) if r>0 else max(0.,ndtr(h)-ndtr(-k))
    def f(x):return np.exp(-x*x/2)/np.sqrt(2*np.pi)*ndtr((k-r*x)/np.sqrt(1-r*r))
    cut=k/r if r else np.inf
    if cut<h:return quad(f,-np.inf,cut,epsabs=2e-13)[0]+quad(f,cut,h,epsabs=2e-13)[0]
    return quad(f,-np.inf,h,epsabs=2e-13)[0]


def test_bvn():
    rng=np.random.default_rng(138)
    hk=np.r_[rng.normal(size=(40,2)),[[0,0],[0,1],[-1,0],[0,-2],[1,0],[-6,-4]]]
    for r in (-1.,-.999,-.7,0.,.7,.999,1.):
        p=bvn(hk[:,0],hk[:,1],r)
        ref=[bvn_reference(h,k,r) for h,k in hk]
        np.testing.assert_allclose(p,ref,atol=3e-10,rtol=0)


def equi(r):
    C=np.full((3,3),r);np.fill_diagonal(C,1.);return C


@pytest.mark.parametrize('r',[-.5,-.49999,.35])
def test_three_contrast(r):
    C=equi(r);mu=np.array([.2,.5,-.3]);ref=[]
    for i in range(3):
        j=np.delete(np.arange(3),i); S=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i];s=np.sqrt(S.diagonal())
        ref.append(bvn_reference(*((mu[j]-mu[i])/s),S[0,1]/np.prod(s)))
    p=field(mu,[C],panels=2048)
    np.testing.assert_allclose(p,ref,atol=4e-9,rtol=0)


def test_four_arcsine():
    s=np.array([.8,1.1,1.7]); C=equi(-.5)*s[:,None]*s[None,:];C=(C+C.T)/2
    blocks=[C,np.array([[.4]])];S=block_diag(*blocks);ref=[]
    for i in range(4):
        j=np.delete(np.arange(4),i);T=S[np.ix_(j,j)]-S[j,i,None]-S[i,j][None,:]+S[i,i];d=np.sqrt(T.diagonal());R=T/d[:,None]/d[None,:]
        ref.append(1/8+np.arcsin(R[np.triu_indices(3,1)]).sum()/(4*np.pi))
    p=field(np.zeros(4),blocks,panels=2048)
    np.testing.assert_allclose(p,ref,atol=2e-9,rtol=0)


def test_invariances():
    blocks=[equi(-.5),equi(.2),np.array([[.9,.4],[.4,.6]]),np.array([[.8]])];m=np.random.default_rng(22).normal(0,.4,9)
    p,info=field(m,blocks,panels=512,return_info=True)
    assert np.all(p>=0) and info['mass_defect']<1e-8
    np.testing.assert_allclose(field(5+3*m,[9*C for C in blocks],panels=512),p,atol=2e-12,rtol=0)
    order=[2,0,3,1];inds=np.split(np.arange(9),[3,6,8]);ix=np.concatenate([inds[j][::-1] for j in order]);b=[blocks[j][::-1,::-1] for j in order]
    np.testing.assert_allclose(field(m[ix],b,panels=512),p[ix],atol=2e-12,rtol=0)


def test_many_exact_singular_and_refusals():
    b=[equi(-.5)]*20;p=field(np.zeros(60),b,panels=256)
    np.testing.assert_allclose(p,np.full(60,1/60),atol=1e-8,rtol=0)
    with pytest.raises(ValueError):field(np.zeros(2),[np.ones((2,2))])
    with pytest.raises(ValueError):field(np.zeros(3),[equi(-.6)])
    with pytest.raises(ValueError):field(np.zeros(3),[np.eye(2)])


def test_factor_and_common_shock():
    from triad_field import factor_field
    b=[equi(-.5),np.array([[.7]])];v=np.array([.3,-.2,.4,-.1]);C=block_diag(*b)+np.outer(v,v);ref=[]
    for i in range(4):
        j=np.delete(np.arange(4),i);T=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i];d=np.sqrt(T.diagonal());R=T/d[:,None]/d[None,:]
        ref.append(1/8+np.arcsin(R[np.triu_indices(3,1)]).sum()/(4*np.pi))
    p=factor_field(np.zeros(4),b,v,order=21,panels=512)
    np.testing.assert_allclose(p,ref,atol=2e-8,rtol=0)
    np.testing.assert_allclose(factor_field(np.zeros(4),b,v+4,order=21,panels=512),p,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(factor_field(np.zeros(4),b,np.ones(4),panels=512),field(np.zeros(4),b,panels=512))
