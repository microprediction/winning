import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from hazard_draw import contrast_split,draw_shares

def test_split():
    rng=np.random.default_rng(4); G=rng.normal(size=(10,10)); C=G@G.T+.2*np.eye(10); A,d=contrast_split(C); P=np.eye(10)-np.ones((10,10))/10
    np.testing.assert_allclose(P@(A@A.T+np.diag(d))@P,P@C@P,atol=1e-12)
    A2,d2=contrast_split(C+7); np.testing.assert_allclose(d2,d,atol=1e-12); np.testing.assert_allclose(A2@A2.T,A@A.T,atol=1e-12)

def test_independent():
    a=np.array([-.4,.2,.7]); s=np.array([.8,1.1,.6]); p=draw_shares(a,np.empty((3,0)),s*s,17,42)
    ref=[]
    for i in range(3):
        j=np.arange(3)!=i
        ref.append(quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)*np.prod(ndtr((a[j]-a[i]-s[i]*z)/s[j])),-12,12,epsabs=1e-13)[0])
    np.testing.assert_allclose(p,ref,atol=2e-5)

def test_equal():
    p=draw_shares(np.zeros(8),np.empty((8,0)),np.ones(8),5,1); np.testing.assert_allclose(p,np.ones(8)/8,atol=1e-15)

def test_cap_clones():
    p=draw_shares(np.array([0.,0.,.2]),np.zeros((3,0)),np.array([0.,0.,1.]),16,19)
    np.testing.assert_allclose(p,[(1-ndtr(-.2))/2]*2+[ndtr(-.2)],atol=2e-5)

def test_singular():
    p=draw_shares(np.zeros(3),np.array([[1.],[1.],[-1.]]),np.zeros(3),10,2)
    np.testing.assert_allclose(p,[.25,.25,.5],atol=1e-15)

def test_common_shift_scale():
    a=np.array([.1,.2,-.3]); A=np.array([[1.,-.2],[.2,.4],[-1.,.3]]); d=np.array([.4,.1,.2]); p=draw_shares(a,A,d,10,9)
    q=draw_shares(4*a+9,4*A+np.array([6.,-9.]),16*d,10,9); np.testing.assert_allclose(p,q,atol=1e-14)
    assert np.all(p>=0) and abs(p.sum()-1)<1e-14

def test_binary():
    C=np.array([[1.,-.4],[-.4,1.5]]); a=np.array([.2,-.5]); A,d=contrast_split(C); p=draw_shares(a,A,d,17,6)
    np.testing.assert_allclose(p[0],ndtr((a[1]-a[0])/np.sqrt(C[0,0]+C[1,1]-2*C[0,1])),atol=2e-5)
