"""Research: rank-two blocks with optional scalar Gaussian observation.
Min wins. No diagonal jitter. Hull pruning optional, not certified.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.special import roots_legendre, wofz


def _tail(z,k):
    p=np.abs(z); q=np.where(z>=0,k,-k)
    f=.5*np.exp(-p*p/2+1j*q*p)*wofz((q+1j*p)/np.sqrt(2))
    return np.where(z>=0,f,np.exp(-k*k/2)-f)


def _interval(lo,hi,k):
    flip=hi<=0
    l=np.where(flip,-hi,lo); h=np.where(flip,-lo,hi); k=np.where(flip,-k,k)
    a=np.where(np.isneginf(l),np.exp(-k*k/2),_tail(np.where(np.isfinite(l),l,0.),k))
    b=np.where(np.isposinf(h),0.,_tail(np.where(np.isfinite(h),h,0.),k))
    return np.where(lo<hi,a-b,0.)


def neighbors(m,R,prune):
    n=len(m)
    def full(): return [np.delete(np.arange(n),i) for i in range(n)]
    if not prune or n<=3: return full()
    # Lift each affine performance to a point in 3D. Constant means
    # reduce to a 2D polygon. No QJ (jitter changes the model).
    p=R if np.all(m==m[0]) else np.column_stack([R,m])
    try: hull=ConvexHull(p,qhull_options='Qc')
    except QhullError: return full()
    if len(hull.coplanar): return full()
    adj=[set() for _ in m]
    for face in hull.simplices:
        for i in face:
            adj[i].update(j for j in face if i!=j)
    return [np.array(sorted(a),int) if a else None for a in adj]


def envelope(a,b):
    """Upper envelope of affine lines. Starts at -inf."""
    order=np.lexsort((b,a)); aa=[]; bb=[]; xx=[]
    for i in order:
        u,v=a[i],b[i]
        if aa and u==aa[-1]:
            aa.pop(); bb.pop(); xx.pop()
        while aa:
            x=(bb[-1]-v)/(u-aa[-1])
            if x>xx[-1]: break
            aa.pop(); bb.pop(); xx.pop()
        if not aa: x=-np.inf
        aa.append(u); bb.append(v); xx.append(x)
    return np.array(aa),np.array(bb),np.array(xx)


def eval_env(env,t):
    a,b,x=env
    if not len(a): return np.full_like(t,-np.inf)
    j=np.searchsorted(x,t,side='right')-1
    return a[j]*t+b[j]


def field(mu,roots,beta,panels=128,nk=32,cutoff=16.,pad=9.,prune=True,batch=8):
    mu=np.asarray(mu,float); beta=np.asarray(beta,float); roots=tuple(np.asarray(r,float) for r in roots)
    if mu.ndim!=1 or beta.shape!=mu.shape or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(beta)) or not roots or sum(len(r) for r in roots)!=len(mu): raise ValueError('invalid input')
    if any(r.ndim!=2 or r.shape[1]!=2 or len(r)<2 or not np.all(np.isfinite(r)) or np.linalg.matrix_rank(r)!=2 for r in roots): raise ValueError('rank-two roots required')
    if not all(isinstance(q,(int,np.integer)) and q>0 for q in (panels,nk,batch)) or not np.all(np.isfinite([pad,cutoff])) or min(pad,cutoff)<=0: raise ValueError('invalid quadrature')
    scale=max(np.linalg.norm(r,axis=1).max() for r in roots); m=(mu-mu.min())/scale; beta=beta*scale
    specs=[]; knots=[]; sd=[]; vs=[]; off=0; edges_count=0; active=0
    for R in roots:
        R=R/scale; M=m[off:off+len(R)]; h=R.T@beta[off:off+len(R)]; s=np.linalg.norm(R,axis=1)
        if np.any(s<=0): raise ValueError('zero marginal')
        # Reject equal/proportional rows to exclude ties and step boundaries.
        # Vectorized row check; avoids storing a b-by-b array.
        angles=np.sort(np.mod(np.arctan2(R[:,1],R[:,0]),np.pi))
        if np.min(np.diff(np.r_[angles,angles[0]+np.pi]))<=64*np.finfo(float).eps: raise ValueError('proportional local rows unsupported')
        adj=neighbors(M,R,prune); group=[]; sd.extend(s); vs.append(h@h)
        for i,j in enumerate(adj):
            if j is None: group.append(None); continue
            edges_count+=len(j); active+=1
            e=R[i]/s[i]; q=np.array([-e[1],e[0]]); v=R[j]@q; reg=R[j]@e/s[i]
            if np.any(v==0): raise ValueError('unresolved local contrast')
            a=(1-reg)/v; b=(reg*M[i]-M[j])/v
            pos=v>0
            low=envelope(a[pos],b[pos]); neg=envelope(-a[~pos],-b[~pos])
            # Only active affine segments matter. Add crossings and local
            # normal-scale thresholds. Empty envelopes are unbounded.
            for A,B,X in (low,neg):
                knots.extend(X[1:]); end=np.r_[X[1:],np.inf]
                for u,w,l,r in zip(A,B,X,end):
                    if u!=0:
                        for z in (-6.,-2.,0.,2.,6.):
                            t=(z-w)/u
                            if l<t<r: knots.append(t)
            if len(low[0]) and len(neg[0]):
                x=np.unique(np.r_[low[2],neg[2],np.inf])
                for l,r in zip(x[:-1],x[1:]):
                    j1=np.searchsorted(low[2],l,side='right')-1; j2=np.searchsorted(neg[2],l,side='right')-1
                    a0=low[0][j1]+neg[0][j2]; b0=low[1][j1]+neg[1][j2]
                    if a0!=0 and l< -b0/a0 <r: knots.append(-b0/a0)
            group.append((low,neg,e@h,q@h))
        specs.append((slice(off,off+len(R)),group)); off+=len(R)
    sd=np.array(sd); vs=np.array(vs); total=vs.sum()
    if total>1+16*np.finfo(float).eps: raise ValueError('observation variance exceeds one')
    adjust=np.sqrt(max(total,1.)); noise=max(0.,1-total)
    if noise<16*np.finfo(float).eps and np.count_nonzero(vs>0)<3: raise ValueError('singular observation needs three active blocks')
    lo=np.min(m-pad*sd); hi=max(np.min(m[sl]+pad*sd[sl]) for sl,_ in specs)
    # Extra marginal-scale knots for heterogeneous performance variances.
    knots=np.r_[knots,(m[:,None]+sd[:,None]*np.array([-6,-2,0,2,6])).ravel()]
    edges=np.unique(np.r_[np.linspace(lo,hi,panels+1),knots[(knots>lo)&(knots<hi)]])
    dx=np.diff(edges); t=np.r_[(edges[:-1,None]+dx[:,None]*np.arange(4)/4).ravel(),edges[-1]]
    zz=(t[:,None]-m)/sd; f=np.exp(-zz*zz/2)/(np.sqrt(2*np.pi)*sd)
    bounds=[]
    for sl,group in specs:
        for spec in group:
            if spec is None: bounds.append(None); continue
            l,u,a,b=spec; bounds.append((eval_env(l,t),-eval_env(u,t),a/adjust,b/adjust))
    if total==0: k=np.array([0.]); kw=np.array([1.]); nk=1
    else:
        k,kw=roots_legendre(nk); k=(k+1)*cutoff/2; kw=kw*cutoff/2*np.sqrt(2/np.pi)*np.exp(-noise*k*k/2)
    p=np.zeros(len(m)); half=np.repeat(dx/2,2)
    for start in range(0,nk,batch):
        slk=slice(start,start+batch); kk=k[slk]; g=np.zeros((len(t),len(kk),len(m)),complex)
        for i,spec in enumerate(bounds):
            if spec is None: continue
            l,u,a,b=spec; good=l<u
            g[good,:,i]=f[good,i,None]*np.exp(1j*zz[good,i,None]*a*kk)*_interval(l[good,None],u[good,None],b*kk)
        G=np.stack([g[:,:,sl].sum(axis=2) for sl,_ in specs],axis=2)
        area=(G[:-2:2]+4*G[1::2]+G[2::2])*half[:,None,None]/6
        S=np.concatenate([np.cumsum(area[::-1],axis=0)[::-1],np.zeros((1,len(kk),len(specs)),complex)],axis=0)
        left=np.ones_like(S); right=np.ones_like(S)
        left[:,:,1:]=np.cumprod(S[:,:,:-1],axis=2); right[:,:,:-1]=np.cumprod(S[:,:,:0:-1],axis=2)[:,:,::-1]
        outer=g[::2].copy()
        for j,(sl,_) in enumerate(specs): outer[:,:,sl]*=(left[:,:,j]*right[:,:,j])[:,:,None]
        integ=np.sum((outer[:-2:2]+4*outer[1::2]+outer[2::2])*dx[:,None,None]/6,axis=0)
        p+=kw[slk]@integ.real
    return p,dict(panels=len(dx),active=active,edges=edges_count,mass_defect=abs(p.sum()-1))
