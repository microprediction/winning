import time,sys
import numpy as np
from scipy.linalg import block_diag
from scipy.stats._qmvnt import _qmvn
from knot_field import field

def decompose(R):
    _,s,v=np.linalg.svd(R[1:]-R[0],full_matrices=True)
    return R@v[:2].T,abs((R@v[2]).mean())

def fixtures(n=24):
    rng=np.random.default_rng(926)
    for kind in ['arbitrary','near']:
        roots=[]; shifts=[]
        for _ in range(n//3):
            if kind=='arbitrary': A=rng.normal(size=(3,3))
            else:
                u,_=np.linalg.qr(rng.normal(size=(3,3))); A=u@np.diag([1.3,.8,.003])
            R,s=decompose(A); roots.append(R); shifts.append(s)
        m=rng.normal(0,.5,n); beta=rng.normal(size=n)
        root=block_diag(*[np.c_[r,np.full(len(r),s)] for r,s in zip(roots,shifts)])
        h=root.T@beta; beta*=np.sqrt((1-1e-8)/(h@h)); C=root@root.T; c=C@beta
        yield kind,(m,roots,shifts,beta,C-np.outer(c,c))

def genz(m,C,budget,seed):
    p=[]; rng=np.random.default_rng(seed)
    for i in range(len(m)):
        j=np.delete(np.arange(len(m)),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        p.append(_qmvn(budget,T,m[i]-m[j],np.full(len(j),np.inf),rng,n_batches=8)[0])
    p=np.array(p); return p/p.sum()

def test_arcsine():
    R=[np.array([[1.,.2],[-.3,.7]]),np.array([[.8,-.4],[.3,.9]])]; s=np.array([.003,.008]); b=np.array([.2,-.4,.1,.35]); m=np.zeros(4)
    root=block_diag(*[np.c_[r,np.full(len(r),v)] for r,v in zip(R,s)]); C=root@root.T; c=C@b; C-=np.outer(c,c)
    ref=[]
    for i in range(4):
        j=np.delete(np.arange(4),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]; d=np.sqrt(T.diagonal()); corr=T/d[:,None]/d[None,:]
        ref.append(.125+np.arcsin(corr[np.triu_indices(3,1)]).sum()/(4*np.pi))
    p=field(m,R,s,b,step=.25,order=8,inner=14,nk=48,cutoff=24)
    print('arcsine',max(abs(p-ref))); np.testing.assert_allclose(p,ref,atol=2e-9,rtol=0)

def test_near():
    _,(m,R,s,b,C)=list(fixtures())[1]
    p=field(m,R,s,b,step=.5,order=6,inner=10,nk=32,cutoff=16)
    q=field(m,R,s,b,step=.25,order=8,inner=14,nk=48,cutoff=24)
    print('near',p.sum()-1,q.sum()-1,max(abs(p-q))); assert p.min()>=-1e-10 and abs(p.sum()-1)<1e-6
    np.testing.assert_allclose(p,q,atol=1e-6,rtol=0)

def test_invariance():
    _,(m,R,s,b,C)=list(fixtures(12))[1]; idx=np.arange(12).reshape(4,3)[::-1,::-1].ravel()
    p=field(m,R,s,b); q=field(7+3*m[idx],[3*r[::-1] for r in R[::-1]],3*np.array(s[::-1]),b[idx]/3)
    np.testing.assert_allclose(p[idx],q,atol=1e-11,rtol=0)

def bench(n,kind):
    _,(m,R,s,b,C)=next((x for x in fixtures(n) if x[0]==kind)); vals=[]
    for step,order,inner,Q,K in [(1.,4,6,24,12),(.5,6,10,32,16),(.35,8,12,40,20),(.25,8,16,48,24)]:
        t=time.perf_counter(); p,d=field(m,R,s,b,step=step,order=order,inner=inner,nk=Q,cutoff=K,diagnostics=True); vals.append(p); print(n,kind,step,order,inner,Q,K,time.perf_counter()-t,d,p.sum()-1,p.min(),None if len(vals)==1 else max(abs(p-vals[-2])),flush=True)
    ref=vals[-1]; print('RMSE',[np.sqrt(np.mean((p-ref)**2)) for p in vals],flush=True)
    for budget in [2048,8192,32768]:
        t=time.perf_counter(); p=np.array([genz(m,C,budget,i+100*(budget==32768)) for i in range(4)]); print('genz',budget,(time.perf_counter()-t)/4,np.sqrt(np.mean((p-ref)**2)),np.sqrt(np.mean(p.var(axis=0,ddof=1))),np.sqrt(np.mean((p.mean(0)-ref)**2)),flush=True)

if __name__=='__main__': bench(int(sys.argv[1]),sys.argv[2])


def test_exact_singular():
    a=np.arange(3)*2*np.pi/3+.1
    R=np.c_[np.cos(a),np.sin(a)]; roots=[R]*4; s=np.zeros(4)
    b=np.tile([.1,-.3,.2],4); b/=np.sqrt(4*np.linalg.norm(R.T@b[:3])**2)
    p=field(np.zeros(12),roots,s,b,step=.3,order=8,inner=12,nk=48,cutoff=24)
    np.testing.assert_allclose(p.reshape(4,3).sum(1),.25,atol=2e-8,rtol=0)
    assert p.min()>0


def test_single_block_shift():
    R=[np.array([[1.,.2],[-.3,.7],[.8,-.4]])]; m=np.array([.2,-.4,.1]); b=np.zeros(3)
    p=field(m,R,[.005],b,step=.25,order=8,inner=14)
    q=field(m,R,[0.],b,step=.25,order=8,inner=14)
    np.testing.assert_allclose(p,q,atol=1e-10,rtol=0)
