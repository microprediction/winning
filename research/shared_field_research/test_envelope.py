import numpy as np
from scipy.special import ndtr
from envelope import scan,prepare,shares

def brute(a,b):
    p=np.zeros(len(a))
    for i in range(len(a)):
        lo,hi=-np.inf,np.inf; ties=1
        for j in range(len(a)):
            if j==i: continue
            d=b[i]-b[j]; e=a[j]-a[i]
            if d>0: hi=min(hi,e/d)
            elif d<0: lo=max(lo,e/d)
            elif e<0: hi=-np.inf
            elif e==0: ties+=1
        if lo<hi: p[i]=(ndtr(hi)-ndtr(lo))/ties
    return p

def test_scan():
    rng=np.random.default_rng(82)
    for n in [1,2,10,50]:
        b=np.sort(rng.integers(-4,5,n).astype(float))[::-1].copy()
        a=rng.integers(-4,5,(100,n)).astype(float)
        expected=sum(brute(x,b) for x in a)
        np.testing.assert_allclose(scan(a,b),expected,atol=2e-13,rtol=0)

def test_rank_one_and_clones():
    a=np.array([-.1,-.1,.5,-.4,2.]); b=np.array([2.,2.,-3.,.1,0.])
    p=shares(prepare(a,b[:,None]))
    np.testing.assert_allclose(p,brute(a,b),atol=1e-14)
    assert p[0]==p[1]

def test_deterministic():
    np.testing.assert_array_equal(shares(prepare([1.,0.,0.],np.ones((3,4)))),[0.,.5,.5])

def test_binary_full_rank():
    a=np.array([-.4,.3]); A=np.array([[1.,0.],[-.2,.8]])
    p=shares(prepare(a,A),m=9)
    np.testing.assert_allclose(p,[ndtr(.7/np.linalg.norm(A[0]-A[1])),ndtr(-.7/np.linalg.norm(A[0]-A[1]))],atol=2e-14)

def test_rank_one_invariance():
    a=np.array([-.1,.5,-.4,2.]); b=np.array([2.,-3.,.1,0.])[:,None]; perm=[2,0,3,1]
    p=shares(prepare(a,b))
    np.testing.assert_allclose(shares(prepare((a[perm]+3)*4,(b[perm]+5)*4)),p[perm],atol=1e-14)

def test_singular_clones_with_residual():
    A=np.array([[1.,2.],[1.,2.],[2.,-3.],[-1.,.4]])
    p=shares(prepare([0.,0.,.5,.2],A),m=10)
    assert p[0]==p[1] and np.all(p>=0) and abs(p.sum()-1)<1e-12
