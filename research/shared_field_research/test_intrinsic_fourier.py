import sys,time
import numpy as np
from scipy.linalg import block_diag
from scipy.stats._qmvnt import _qmvn
from scipy.integrate import quad
from intrinsic_fourier import field, interval_cf, endpoint_boundary
from scipy.stats import qmc
from scipy.special import ndtri


def fixture(n,kind):
    rng=np.random.default_rng(184); R=[]
    for i in range(n//3):
        theta=np.arange(3)*2*np.pi/3
        B=np.c_[np.cos(theta),np.sin(theta)]
        if kind!='exact':
            if kind=='mixed': B=np.linalg.cholesky(.5*np.eye(3)+B@B.T)
            else: B=np.c_[B*np.sqrt(1-1e-5),np.ones(3)*np.sqrt(1e-5)]
        if kind=='hetero': B=B*rng.uniform(.5,1.5,3)[:,None]
        R.append(B)
    C=block_diag(*(B@B.T for B in R)); b=rng.normal(size=n); b*=np.sqrt((1-1e-5)/(b@C@b)); m=rng.normal(0,.4,n)
    return m,R,b,C-np.outer(C@b,C@b)


def genz(m,C,budget,seed):
    p=[]; rng=np.random.default_rng(seed)
    for i in range(len(m)):
        j=np.delete(np.arange(len(m)),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        p.append(_qmvn(budget,T,m[i]-m[j],np.full(len(j),np.inf),rng,n_batches=8)[0])
    p=np.array(p); return p/p.sum()


def test_interval():
    for l,h in [(-np.inf,-1),(-2,-.5),(-1,1),(1,2),(0,np.inf),(-np.inf,np.inf)]:
        k=2.1; ref=quad(lambda z:np.exp(-z*z/2+1j*k*z)/np.sqrt(2*np.pi),l,h,complex_func=True,epsabs=1e-12)[0]
        np.testing.assert_allclose(interval_cf(np.array(l),np.array(h),k),ref,atol=1e-12)


def test_exact_arcsine():
    m,R,b,C=fixture(3,'exact'); R.append(np.ones((1,1))); m=np.zeros(4); b=np.r_[b*.5,.3]; C=block_diag(*(B@B.T for B in R)); C=C-np.outer(C@b,C@b)
    p=[]
    for i in range(4):
        j=np.delete(np.arange(4),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]; s=np.sqrt(np.diag(T)); corr=T/np.outer(s,s)
        p.append(.125+np.arcsin(corr[np.triu_indices(3,1)]).sum()/(4*np.pi))
    np.testing.assert_allclose(field(m,R,b,panels=512,nk=48,cutoff=20,batch=48),p,atol=5e-9,rtol=0)


def test_singular():
    R=[np.array([[1.,0.],[-.5,np.sqrt(3)/2],[-.5,-np.sqrt(3)/2]])]*4
    b=np.tile([1.,0.,0.],4)/2
    p=field(np.zeros(12),R,b,panels=192,nk=40,cutoff=20,batch=40)
    assert p.min()>0 and abs(p.sum()-1)<2e-6
    np.testing.assert_allclose(p.reshape(4,3),np.tile(p[:3],(4,1)),atol=1e-10)
    np.testing.assert_allclose(p[1],p[2],atol=1e-10)


def test_invariance():
    m,R,b,C=fixture(12,'exact'); kw=dict(panels=128,nk=32,cutoff=16,batch=32)
    p=field(m,R,b,**kw); idx=np.arange(12).reshape(-1,3)[::-1,::-1].ravel()
    q=field(3*m[idx]+7,[3*x[::-1] for x in R[::-1]],b[idx]/3,**kw)
    np.testing.assert_allclose(p[idx],q,atol=1e-11); assert p.min()>0 and abs(p.sum()-1)<2e-6


def test_endpoint():
    v=np.array([1.,-.9]); w=np.array([.01,.012]); d=np.array([[-.3,.27],[.1,-.09]])
    a,b=.4,.01; k=np.array([0.,2.,8.]); out=endpoint_boundary(d,v,w,a,b,k,64)
    for i in range(len(d)):
        for j in range(len(k)):
            def f(y):
                lo=(d[i,0]-w[0]*y)/v[0]; hi=(d[i,1]-w[1]*y)/v[1]
                return np.exp(-y*y/2+1j*b*y*k[j])/np.sqrt(2*np.pi)*interval_cf(lo,hi,a*k[j])
            ref=quad(f,-9,9,complex_func=True,epsabs=1e-13,points=[0])[0]
            np.testing.assert_allclose(out[i,j],ref,atol=3e-9,rtol=0)


if __name__=='__main__':
    n=int(sys.argv[1]);kind=sys.argv[2];m,R,b,C=fixture(n,kind)
    if '--root-reference' not in sys.argv:
        ref=np.array([genz(m,C,32768,100+i) for i in range(4)])
        print('reference SD',np.sqrt(np.mean(ref.var(axis=0,ddof=1))),flush=True);ref=ref.mean(axis=0)
        for budget in [2048,8192]:
            t=time.perf_counter();p=np.array([genz(m,C,budget,i) for i in range(4)])
            print('genz',budget,(time.perf_counter()-t)/4,np.sqrt(np.mean((p-ref)**2)),np.sqrt(np.mean(p.var(axis=0,ddof=1))),flush=True)
    else:
        root=block_diag(*R); h=root.T@b; noise=max(0.,1-h@h)
        def draw(seed):
            z=ndtri(qmc.Sobol(root.shape[1]+1,scramble=True,seed=seed).random_base2(18)); y=z[:,:-1]
            y=y-(y@h+np.sqrt(noise)*z[:,-1])[:,None]*h
            return np.bincount(np.argmin(y@root.T+m,axis=1),minlength=n)/len(z)
        ref=np.array([draw(i) for i in range(4)]); print("root reference SD",np.sqrt(np.mean(ref.var(axis=0,ddof=1))),flush=True); ref=ref.mean(axis=0)
    prev=None
    settings=[(64,24,12,4),(128,32,16,6)] if kind=="hard" else [(96,28,14,6),(192,40,20,8),(384,56,28,8)]
    for L,Q,K,O in settings:
        t=time.perf_counter();p,info=field(m,R,b,panels=L,nk=Q,cutoff=K,order=O,batch=Q,return_info=True)
        print('field',L,Q,K,O,time.perf_counter()-t,info,'min',p.min(),'refine',None if prev is None else np.max(abs(p-prev)),'rmse',None if ref is None else np.sqrt(np.mean((p-ref)**2)),flush=True);prev=p
