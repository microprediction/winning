import sys,time
import numpy as np
from scipy.linalg import block_diag
from scipy.special import ndtr,ndtri
from scipy.integrate import quad
from scipy.stats import qmc
from scipy.stats._qmvnt import _qmvn
from planar_field import field,neighbors,envelope,eval_env


def fixture(n,b,kind='mixed'):
    rng=np.random.default_rng(315); R=[]
    for i in range(n//b):
        theta=np.arange(b)*2*np.pi/b+rng.uniform(-.15,.15,b)/b
        r=np.c_[np.cos(theta),np.sin(theta)]
        if kind!='circle': r*=rng.uniform(.5,1.5,(b,1))
        R.append(r)
    m=rng.normal(0,.45,n); beta=rng.normal(size=n)
    root=block_diag(*R); h=root.T@beta
    beta*=np.sqrt((1-1e-8)/(h@h)); C=root@root.T; c=C@beta
    return m,R,beta,C-np.outer(c,c)


def independent_cell(m,R,i):
    # Integrate along a fixed Cartesian latent axis, not performance t.
    d=R-R[i]; a=m[i]-m; mask=np.arange(len(m))!=i
    d=d[mask]; a=a[mask]; x=[]
    for j in range(len(a)):
        if d[j,1]==0 and d[j,0]!=0: x.append(a[j]/d[j,0])
        for k in range(j):
            den=d[j,0]*d[k,1]-d[k,0]*d[j,1]
            if den!=0: x.append((a[j]*d[k,1]-a[k]*d[j,1])/den)
    def f(z):
        lo=-np.inf; hi=np.inf
        for (u,v),w in zip(d,a):
            if v>0: lo=max(lo,(w-u*z)/v)
            elif v<0: hi=min(hi,(w-u*z)/v)
            elif u*z<=w: return 0.
        return max(0.,ndtr(hi)-ndtr(lo))*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    edges=np.unique(np.r_[-10.,[z for z in x if -10<z<10],10.]);return sum(quad(f,l,u,epsabs=1e-13)[0] for l,u in zip(edges[:-1],edges[1:]))


def genz(m,C,budget,seed):
    p=[]; rng=np.random.default_rng(seed)
    for i in range(len(m)):
        j=np.delete(np.arange(len(m)),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        p.append(_qmvn(budget,T,m[i]-m[j],np.full(len(j),np.inf),rng,n_batches=8)[0])
    return np.array(p)


def root_sample(m,R,beta,pow2,seed):
    root=block_diag(*R); h=root.T@beta; noise=max(0.,1-h@h); n=len(m)
    z=ndtri(qmc.Sobol(len(h)+1,scramble=True,seed=seed).random_base2(pow2)); counts=np.zeros(n)
    for x in np.array_split(z,max(1,len(z)//16384)):
        y=x[:,:-1]; y=y-(y@h+np.sqrt(noise)*x[:,-1])[:,None]*h
        counts+=np.bincount(np.argmin(y@root.T+m,axis=1),minlength=n)
    return counts/len(z)


def test_envelopes():
    rng=np.random.default_rng(21); a=rng.integers(-3,4,100).astype(float); b=rng.normal(size=100); t=np.linspace(-20,20,500)
    np.testing.assert_allclose(eval_env(envelope(a,b),t),np.max(a*t[:,None]+b,axis=1),atol=2e-14)


def test_cell():
    m,R,_,_=fixture(13,13); p,info=field(m,R,np.zeros(13),panels=384); ref=np.array([independent_cell(m,R[0],i) for i in range(13)])
    np.testing.assert_allclose(p,ref,atol=3e-8,rtol=0); assert p.min()>=0 and abs(p.sum()-1)<1e-7


def test_global():
    m,R,beta,C=fixture(24,8); opts=dict(panels=192,nk=40,cutoff=20,batch=8)
    p,info=field(m,R,beta,**opts); q,_=field(m,R,beta,prune=False,**opts)
    np.testing.assert_allclose(p,q,atol=3e-8,rtol=0); assert p.min()>=-1e-12 and abs(p.sum()-1)<1e-7
    idx=np.arange(24).reshape(-1,8)[::-1,::-1].ravel(); q,_=field(4*m[idx]+7,[4*x[::-1] for x in R[::-1]],beta[idx]/4,**opts)
    np.testing.assert_allclose(q,p[idx],atol=2e-12,rtol=0)
    # Explicit singular observation, same original model not a ridge.
    root=block_diag(*R); beta=beta/np.linalg.norm(root.T@beta); q,_=field(m,R,beta,**opts)
    assert abs(q.sum()-1)<1e-7 and np.max(abs(p-q))<1e-8


def test_regular():
    # Odd polygon avoids proportional antipodal rows. Equal means enable 2D hull.
    b=11; theta=np.arange(b)*2*np.pi/b; R=[np.c_[np.cos(theta),np.sin(theta)]]*3
    p,info=field(np.zeros(3*b),R,np.zeros(3*b),panels=256)
    np.testing.assert_allclose(p,np.full(3*b,1/(3*b)),atol=1e-8,rtol=0)
    assert info['edges']==2*3*b


def test_bad():
    import pytest
    with pytest.raises(ValueError): field([0,0,0],[np.array([[1.,0.],[2.,0.],[0.,1.]])],[0,0,0])
    with pytest.raises(ValueError): field([0,0],[np.eye(2)],[2,0])


def bench(n,b):
    m,R,beta,C=fixture(n,b); ref=None
    for L,Q,K in [(96,24,12),(192,40,20),(384,56,28)]:
        t=time.perf_counter(); p,info=field(m,R,beta,panels=L,nk=Q,cutoff=K,batch=8); elapsed=time.perf_counter()-t
        print('field',n,b,L,Q,K,elapsed,info,'min',p.min(),'refine',None if ref is None else np.max(abs(p-ref)),flush=True); ref=p
    t=time.perf_counter(); q,info=field(m,R,beta,panels=192,nk=40,cutoff=20,batch=8,prune=False); print('allpairs',time.perf_counter()-t,info,'maxerr',np.max(abs(q-ref)),flush=True)
    for budget in [2048,8192]:
        t=time.perf_counter(); p=np.array([genz(m,C,budget,i) for i in range(4)]); elapsed=(time.perf_counter()-t)/4
        mass=np.max(abs(p.sum(axis=1)-1)); p=p/p.sum(axis=1,keepdims=True); print('genz',budget,elapsed,'raw mass defect',mass,'rmse',np.sqrt(np.mean((p-ref)**2)),'SD',np.sqrt(np.mean(p.var(axis=0,ddof=1))),'mean RMSE',np.sqrt(np.mean((p.mean(axis=0)-ref)**2)),flush=True)
    t=time.perf_counter(); p=np.array([root_sample(m,R,beta,19,i+100) for i in range(4)]); print('root reference',(time.perf_counter()-t)/4,'mean rmse',np.sqrt(np.mean((p.mean(axis=0)-ref)**2)),'SD',np.sqrt(np.mean(p.var(axis=0,ddof=1))),'max',np.max(abs(p.mean(axis=0)-ref)),flush=True)


def polygons():
    for b in [101,301]:
        theta=np.arange(b)*2*np.pi/b; R=np.c_[np.cos(theta),np.sin(theta)]
        for prune in [True,False]:
            t=time.perf_counter();p,info=field(np.zeros(b),[R],np.zeros(b),panels=96,prune=prune)
            print('polygon',b,prune,time.perf_counter()-t,info,max(abs(p-1/b)),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='polygons': polygons()
    else: bench(int(sys.argv[1]),int(sys.argv[2]))
