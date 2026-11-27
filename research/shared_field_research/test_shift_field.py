import numpy as np
import time,sys
from scipy.linalg import block_diag
from scipy.special import ndtr,ndtri
from scipy.stats import qmc
from scipy.stats._qmvnt import _qmvn
from shift_field import field,decompose,shifted


def fixture(n,b,sigma):
    rng=np.random.default_rng(825); R=[]
    for _ in range(n//b):
        a=np.arange(b)*2*np.pi/b+rng.uniform(-.12,.12,b)/b
        R.append(np.c_[np.cos(a),np.sin(a)]*rng.uniform(.6,1.4,(b,1)))
    m=rng.normal(0,.5,n); s=sigma*rng.uniform(.8,1.2,n//b); beta=rng.normal(size=n)
    root=block_diag(*[np.c_[r,np.ones(len(r))*v] for r,v in zip(R,s)]); h=root.T@beta; beta*=np.sqrt((1-1e-8)/(h@h)); C=root@root.T; c=C@beta
    return m,R,s,beta,C-np.outer(c,c),root


def genz(m,C,budget,seed):
    p=[]; rng=np.random.default_rng(seed)
    for i in range(len(m)):
        j=np.delete(np.arange(len(m)),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        p.append(_qmvn(budget,T,m[i]-m[j],np.full(len(j),np.inf),rng,n_batches=8)[0])
    p=np.array(p); return p/p.sum()


def test_decompose():
    rng=np.random.default_rng(48); root=rng.normal(size=(3,3)); R,s=decompose(root)
    np.testing.assert_allclose(R@R.T+s*s,root@root.T,atol=1e-14)


def test_multiplier():
    from scipy.fft import fftfreq
    t=np.arange(2048)*.02-20.48; k=np.array([0.,1.,4.]); s=.7; h=-.6; a=.4
    g=np.exp(-t[:,None]**2/2+1j*t[:,None]*a*k)/np.sqrt(2*np.pi)
    y=shifted(g,2*np.pi*fftfreq(len(t),.02),k,s,h)
    var=1+s*s; c=a+s*h; v=a*a+h*h-c*c/var
    ref=np.exp(-t[:,None]**2/(2*var)+1j*t[:,None]*k*c/var-.5*k*k*v)/np.sqrt(2*np.pi*var)
    np.testing.assert_allclose(y,ref,atol=3e-15)


def test_single_block_shift():
    m,R,s,b,C,_=fixture(9,9,.5); b*=0
    p=field(m,R,s,b,M=8192); q=field(m,R,[0],b,M=8192)
    np.testing.assert_allclose(p,q,atol=2e-6); assert abs(p.sum()-1)<2e-6


def test_symmetry():
    a=np.arange(3)*2*np.pi/3; R=np.c_[np.cos(a),np.sin(a)]; roots=[R]*4; s=np.ones(4)*.2; b=np.tile([.1,-.3,.2],4); b/=np.sqrt(4*np.linalg.norm(R.T@b[:3])**2)
    p=field(np.zeros(12),roots,s,b,M=8192,nk=48,cutoff=24)
    # identical blocks but contestant means/observation loadings differ;
    # blocks must have equal mass; rotating all input order only permutes output.
    np.testing.assert_allclose(p.reshape(4,3).sum(1),.25,atol=2e-6)
    idx=np.arange(12).reshape(4,3)[::-1,::-1].ravel()
    q=field(np.full(12,7.),[3*R[::-1]]*4,3*s,b[idx]/3,M=8192,nk=48,cutoff=24)
    np.testing.assert_allclose(q,p[idx],atol=1e-13)


def test_arcsine():
    # Four contestants, arbitrary correlated pairs and noisy global score.
    R=[np.array([[1.,.2],[-.3,.7]]),np.array([[.8,-.4],[.3,.9]])]
    s=np.array([.3,.8]); b=np.array([.2,-.4,.1,.35]); m=np.zeros(4)
    root=block_diag(*[np.c_[r,np.full(len(r),v)] for r,v in zip(R,s)])
    C=root@root.T; c=C@b; C-=np.outer(c,c)
    ref=[]
    for i in range(4):
        j=np.delete(np.arange(4),i); T=C[np.ix_(j,j)]-C[j,i,None]-C[None,i,j]+C[i,i]
        d=np.sqrt(T.diagonal()); corr=T/d[:,None]/d[None,:]
        ref.append(.125+np.arcsin(corr[np.triu_indices(3,1)]).sum()/(4*np.pi))
    p=field(m,R,s,b,M=2048,nk=48,cutoff=24)
    np.testing.assert_allclose(p,ref,atol=2e-10,rtol=0)


def bench(n,b,sigma):
    m,R,s,beta,C,root=fixture(n,b,sigma); ref=None; vals=[]
    for M,Q,K in [(512,24,12),(1024,32,16),(2048,40,20),(4096,48,24),(8192,56,28)]:
        t=time.perf_counter(); p=field(m,R,s,beta,M=M,nk=Q,cutoff=K); print('field',M,Q,K,time.perf_counter()-t,'mass',p.sum()-1,'min',p.min(),'diff',None if ref is None else max(abs(p-ref)),flush=True); ref=p; vals.append(p)
    print("field RMSE",[np.sqrt(np.mean((v-ref)**2)) for v in vals],"max relative",[np.max(abs(v-ref)[ref>1e-4]/ref[ref>1e-4]) for v in vals],flush=True)
    for budget in [2048,8192,32768]:
        t=time.perf_counter(); p=np.array([genz(m,C,budget,i+100*(budget==32768)) for i in range(4)]); print('genz',budget,(time.perf_counter()-t)/4,'rmse',np.sqrt(np.mean((p-ref)**2)),'SD',np.sqrt(np.mean(p.var(axis=0,ddof=1))),'mean rmse',np.sqrt(np.mean((p.mean(0)-ref)**2)),flush=True)
    h=root.T@beta; t=time.perf_counter(); ps=[]
    for seed in range(4):
        z=ndtri(qmc.Sobol(len(h)+1,scramble=True,seed=seed+100).random_base2(18)); y=z[:,:-1]; y=y-(y@h+np.sqrt(max(0,1-h@h))*z[:,-1])[:,None]*h
        ps.append(np.bincount(np.argmin(y@root.T+m,axis=1),minlength=n)/len(y))
    p=np.array(ps); print('root',(time.perf_counter()-t)/4,'SD',np.sqrt(np.mean(p.var(axis=0,ddof=1))),'mean rmse',np.sqrt(np.mean((p.mean(0)-ref)**2)),flush=True)

if __name__=='__main__': bench(int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]))
