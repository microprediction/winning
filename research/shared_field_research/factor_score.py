"""Research-only: conditional factor covariance-score remainder.

Exact on SPD contrasts (up to quadrature and sampling); not a production API.
Run --test or --bench from this directory, with single-thread BLAS.
"""
import numpy as np
from scipy.special import log_ndtr, ndtri, ndtri_exp, logsumexp, roots_hermitenorm
from scipy.integrate import quad_vec
from scipy.stats import qmc
from scipy.linalg import null_space
from score_field import moments

def prepare(C, rank=1, sweeps=12):
    C=np.asarray(C,float); n=len(C)
    if n<3 or C.shape!=(n,n) or not np.isfinite(C).all() or not np.allclose(C,C.T):
        raise ValueError('finite symmetric covariance, n >= 3')
    if rank<0 or rank>=n-1 or sweeps<0:
        raise ValueError('invalid rank or sweeps')
    P=np.eye(n)-np.ones((n,n))/n; A=P@C@P
    floor=np.trace(A)/n*1e-6
    if floor<=0: raise ValueError('positive contrast variance required')
    def fitdiag(A):
        return np.maximum((np.diag(A)-np.trace(A)/(n*(n-1)))/(1-2/n),floor)
    d=fitdiag(A)
    for k in range(sweeps+1):
        s=np.sqrt(d); w=1/s; w/=np.linalg.norm(w)
        B=null_space(w[None,:]); T=B.T@(A/s[:,None]/s[None,:])@B
        ev,U=np.linalg.eigh(T)
        if ev[0]<=1e-13*ev[-1]:
            raise ValueError('SPD on contrasts required; use exact-root fallback')
        Awhite=B@T@B.T+np.outer(w,w)
        Q=B@U; bulk=np.median(ev[:n-1-rank]); idx=np.arange(n-1-rank,n-1)
        load=Q[:,idx]*np.sqrt(np.maximum(ev[idx]-bulk,0)); V=s[:,None]*load
        E=Awhite-load@load.T; E=E*s[:,None]*s[None,:]
        sd=np.sqrt(np.diag(E)); R=E/sd[:,None]/sd[None,:]
        if k<sweeps: d=.5*d+.5*fitdiag(P@E@P)
    K=R-np.eye(n); J=np.eye(n)-np.linalg.solve(R,np.eye(n)); c=-.5*np.linalg.slogdet(R)[1]
    return V,sd,K,J,c,np.linalg.eigvalsh(R)

def field(mu,prep,order=21,tol=1e-10):
    V,sd,K,*_=prep; mu=np.asarray(mu,float); rank=V.shape[1]
    nodes,weights=roots_hermitenorm(order); weights/=np.sqrt(2*np.pi)
    if rank:
        fs=np.array(np.meshgrid(*([nodes]*rank),indexing='ij')).reshape(rank,-1).T
        wt=np.prod(np.array(np.meshgrid(*([weights]*rank),indexing='ij')),axis=0).ravel()
    else: fs=np.zeros((1,0)); wt=np.ones(1)
    means=mu+fs@V.T
    def f(t):
        z=(t-means)/sd; ls=log_ndtr(-z)
        w=np.exp(ls.sum(axis=1,keepdims=True)-ls-.5*z*z-np.log(sd*np.sqrt(2*np.pi)))
        return np.stack((wt@w,wt@(w*moments(z,K))))
    out=np.zeros((2,len(mu)))
    bounds=np.unique(np.r_[-np.inf,(means-8*sd).min(),mu.min(),mu.max(),(means+8*sd).max(),np.inf])
    for l,r in zip(bounds[:-1],bounds[1:]):
        out+=quad_vec(f,l,r,epsabs=tol,epsrel=tol)[0]
    return out

def sample(mu,prep,p0,dp,m,seed,batch=4096):
    V,sd,K,J,c,ev=prep; n=len(mu); rank=V.shape[1]
    sob=qmc.Sobol(2*n+rank,scramble=True,seed=seed)
    total=np.zeros((3,n)); ss=np.zeros(3); p0n=p0/p0.sum()
    for k in range(0,2**m,batch):
        u=sob.random(min(batch,2**m-k)); means=mu+ndtri(u[:,:rank])@V.T
        t=(means+sd*ndtri(u[:,rank:rank+n])).min(1)
        z=(t[:,None]-means)/sd; ls=log_ndtr(-z)
        lh=-.5*z*z-ls-np.log(sd); h=np.exp(lh-logsumexp(lh,axis=1,keepdims=True))
        y=-ndtri_exp(np.log1p(-u[:,rank+n:])+ls); delta=z-y
        jy=y@J; ky=y@K
        lr=c+.5*np.sum(y*jy,axis=1,keepdims=True)+delta*jy+.5*delta*delta*np.diag(J)
        sc=.5*(np.sum(y*ky,axis=1,keepdims=True)-np.trace(K))+delta*ky+.5*delta*delta*np.diag(K)
        r=np.expm1(lr); a=h*r; b=h*(r-sc)
        v=np.stack((a,b,b-b.sum(axis=1,keepdims=True)*p0n))
        total+=v.sum(1); ss+=np.sum(v*v,axis=(1,2))
    total/=2**m
    base=p0+dp; base=base+(1-base.sum())*p0n
    return np.stack((p0+total[0],p0+dp+total[1],base+total[2])),ss/2**m-np.sum(total*total,axis=1)

def fixture(n,case):
    rng=np.random.default_rng(812); mu=rng.normal(0,.4,n)
    G=rng.normal(size=(n,n)); W=G@G.T/n; C=np.eye(n)+.04*W
    if 'spike' in case:
        v=rng.normal(0,.65,n); C+=np.outer(v,v)
    if 'hetero' in case:
        s=np.geomspace(.1,1.5,n); C=C*s[:,None]*s[None,:]
    if case=='wishart': C=W+.05*np.eye(n)
    if case=='diffuse':
        u=rng.normal(size=n); u-=u.mean(); u/=np.linalg.norm(u)
        C=np.eye(n)-.99999*np.outer(u,u)
    return mu,C

def test():
    def reconstruct(a):
        V,sd,K,*_=a
        return V@V.T+sd[:,None]*(np.eye(len(sd))+K)*sd[None,:]
    for case in ['spike','heterospike','wishart','diffuse']:
        n=32; mu,C=fixture(n,case); P=np.eye(n)-np.ones((n,n))/n
        a=prepare(C)
        np.testing.assert_allclose(P@reconstruct(a)@P,P@C@P,atol=1e-12)
        b=prepare(9*C+7)
        np.testing.assert_allclose(field(3*mu+4,b),field(mu,a),atol=1e-10)
        perm=np.random.default_rng(42).permutation(n)
        b=prepare(C[np.ix_(perm,perm)])
        np.testing.assert_allclose(field(mu[perm],b),field(mu,a)[:,perm],atol=1e-10)
    # Independent exact reference: zero-mean 3D orthant arcsine formula.
    C=np.array([[1,.13,-.12,.04],[.13,1,.1,-.06],[-.12,.1,1,.16],[.04,-.06,.16,1]])
    p=[]
    for i in range(4):
        j=np.delete(np.arange(4),i)
        A=C[np.ix_(j,j)]-C[j,i,None]-C[i,j][None,:]+C[i,i]
        s=np.sqrt(np.diag(A)); R=A/s[:,None]/s[None,:]
        p.append(1/8+np.arcsin(R[np.triu_indices(3,1)]).sum()/(4*np.pi))
    mu=np.zeros(4); a=prepare(C); p0,dp=field(mu,a,41)
    est,var=sample(mu,a,p0,dp,16,41)
    np.testing.assert_allclose(est[2],p,atol=2e-5)
    assert abs(est[2].sum()-1)<1e-14 and np.min(est[2])>=0
    assert np.linalg.norm(prepare(fixture(32,'heterospike')[1])[2])<.3
    try: prepare(np.ones((3,3)))
    except ValueError: pass
    else: raise AssertionError('singular contrast not refused')
    print('reconstruction, field invariances, exact orthant, mass and singular refusal passed')

def bench():
    from bench_coupled import ghk
    from time import perf_counter
    def g(mu,C,m,s):
        p=ghk(mu,C,m,s); return p/p.sum()
    for n,case in [(32,'spike'),(64,'spike'),(32,'heterospike'),(32,'hetero'),(32,'wishart'),(32,'diffuse')]:
        mu,C=fixture(n,case)
        ref=np.array([g(mu,C,13 if n==32 else 12,s) for s in range(100,104)]); r=ref.mean(0)
        print(n,case,'refsd',np.sqrt(np.mean(np.var(ref,axis=0,ddof=1))),flush=True)
        def report(p):
            return [float(x) for x in (np.sqrt(np.mean((p-r)**2)),np.sqrt(np.mean(np.var(p,axis=0,ddof=1))),np.median(np.max(abs(p-r),axis=1)),np.max(abs(p.sum(1)-1)),p.min())]
        t=perf_counter(); p=np.array([g(mu,C,12,s) for s in range(4)])
        print('ghk',(perf_counter()-t)/4,report(p),flush=True)
        for sweeps in [0,12]:
            t=perf_counter(); a=prepare(C,1,sweeps); p0,dp=field(mu,a); pre=perf_counter()-t
            fine=field(mu,a,41)
            print('sweeps',sweeps,'prep',pre,'eig',a[-1][[0,-1]],'field diff',np.max(abs(fine-np.array([p0,dp]))),'linear rmse',np.sqrt(np.mean((p0+dp-r)**2)),flush=True)
            for m in ([12,14,16] if sweeps and case not in ['wishart','diffuse'] else [14]):
                t=perf_counter(); res=[sample(mu,a,p0,dp,m,s) for s in range(4)]
                dt=(perf_counter()-t)/4; ps=np.array([x[0] for x in res]); vs=np.mean([x[1] for x in res],axis=0)
                print('m',m,'sec',dt,'var',vs,'LR1',report(ps[:,1]),'masscv',report(ps[:,2]),flush=True)

if __name__=='__main__':
    import sys
    if '--test' in sys.argv: test()
    elif '--bench' in sys.argv: bench()
