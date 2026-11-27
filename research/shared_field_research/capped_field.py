"""Conditional minimum-quantile shared field, research prototype (min-wins)."""
import numpy as np
from scipy.special import log_ndtr, ndtri, logsumexp, roots_legendre
from scipy.stats import qmc

def quantile_credit(a,s,u):
    if len(s)==1: return np.ones_like(a)
    n=len(s); lo=np.min(a+ndtri(u[:,None]/n)*s,axis=1); hi=np.min(a+ndtri(u[:,None])*s,axis=1)
    target=np.log1p(-u); x=(lo+hi)/2
    for _ in range(64):
        z=(x[:,None]-a)/s; logs=log_ndtr(-z); r=logs.sum(1)-target
        lh=-.5*z*z-.5*np.log(2*np.pi)-np.log(s)-logs; h=logsumexp(lh,axis=1)
        done=abs(r)<=2e-12*abs(target)
        if np.all(done): break
        lo=np.where((r>0)&~done,x,lo); hi=np.where((r<0)&~done,x,hi)
        proposal=x+r*np.exp(-h)
        x=np.where(done,x,np.where((proposal>lo)&(proposal<hi),proposal,(lo+hi)/2))
    else:
        if np.any(abs(r)>1e-8*abs(target)): raise FloatingPointError('unresolved quantile')
    return np.exp(lh-h[:,None])

def capped_shares(mu,A,d,m=10,L=1,seed=0,batch=256):
    """Exact factor law, zero d = deterministic conditional cap. Equal tie credit.

    L=1 uses one randomized minimum quantile; L>=2 is a positive Legendre
    rule. m controls outer Sobol count (2**m). No post-normalization.
    """
    mu,A,d=map(lambda x:np.asarray(x,float),(mu,A,d)); n=len(mu)
    if mu.shape!=(n,) or n==0 or A.ndim!=2 or A.shape[0]!=n or d.shape!=(n,) or not all(np.isfinite(x).all() for x in (mu,A,d)) or np.any(d<0): raise ValueError('invalid Gaussian factor law')
    if not all(isinstance(x,(int,np.integer)) and x>=0 for x in (m,L,batch)) or min(L,batch)<1: raise ValueError('invalid resolution')
    A=A-A[:1]; mu=mu-mu[0]; scale=max(np.max(abs(A),initial=0),np.sqrt(d).max(),np.max(abs(mu)))
    if scale==0: return np.ones(n)/n
    if not np.isfinite(scale): raise FloatingPointError("contrast range exceeds prototype arithmetic")
    A=A/scale; mu=mu/scale; s=np.sqrt(d)/scale
    if np.any((d>0)&(s==0)): raise FloatingPointError('scale underflow')
    leaf=s>0; core=~leaf; k=A.shape[1]; dim=k+(L==1)
    u=np.clip(qmc.Sobol(dim,scramble=True,seed=seed).random_base2(m),np.finfo(float).eps,1-np.finfo(float).eps) if dim else np.empty((1,0))
    if L>1: q,w=roots_legendre(L); q=(q+1)/2; w=w/2
    out=np.zeros(n)
    for start in range(0,len(u),batch):
        v=u[start:start+batch]; a=mu+ndtri(v[:,:k])@A.T
        if np.any(core):
            c=a[:,core]; cap=c.min(1); ties=c==cap[:,None]
            logS=log_ndtr((a[:,leaf]-cap[:,None])/s[leaf]).sum(1); mass=-np.expm1(logS)
            out[core]+=np.sum(ties/ties.sum(1)[:,None]*np.exp(logS)[:,None],axis=0)
        else: mass=np.ones(len(a))
        active=mass>0
        if not np.any(leaf) or not np.any(active): continue
        aa=a[active][:,leaf]; mm=mass[active]
        rule=[(v[active,-1],1.)] if L==1 else zip(q,w)
        for t,weight in rule:
            level=np.clip(mm*t,np.finfo(float).tiny,np.nextafter(1.,0.))
            credit=quantile_credit(aa,s[leaf],level)
            out[leaf]+=weight*np.sum(mm[:,None]*credit,axis=0)
    return out/len(u)


def precision_factor(Q, leaves):
    """Exact SPD precision to conditional-independent leaf law; no dropped edges.

    Leaves must be an independent set of the precision graph. Dense core
    Schur complement and Cholesky; cubic in core size. Explicit output.
    """
    from scipy.linalg import solve_triangular
    Q=np.asarray(Q,float); n=len(Q); leaves=np.asarray(leaves)
    if Q.shape!=(n,n) or n==0 or not np.isfinite(Q).all() or not np.array_equal(Q,Q.T): raise ValueError('finite symmetric precision required')
    if leaves.ndim!=1 or leaves.dtype.kind not in 'iu' or len(np.unique(leaves))!=len(leaves) or np.any((leaves<0)|(leaves>=n)): raise ValueError('invalid leaf indices')
    core=np.setdiff1d(np.arange(n),leaves); block=Q[np.ix_(leaves,leaves)]; diag=np.diag(block)
    if np.any(diag<=0) or np.any(block!=np.diag(diag)): raise ValueError('leaves must be an independent set with positive pivots')
    B=-Q[np.ix_(leaves,core)]/diag[:,None]; S=Q[np.ix_(core,core)]+Q[np.ix_(core,leaves)]@B
    R=solve_triangular(np.linalg.cholesky((S+S.T)/2).T,np.eye(len(core))) if len(core) else np.empty((0,0))
    A=np.zeros((n,len(core))); A[core]=R; A[leaves]=B@R
    d=np.zeros(n); d[leaves]=1/diag
    return A,d
