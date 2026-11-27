"""Research-only signed block field, exact local roots of rank <= 3."""
import numpy as np
from scipy.special import roots_legendre, wofz


def tail(z,k):
    p=np.abs(z); q=np.where(z>=0,k,-k)
    f=.5*np.exp(-p*p/2+1j*q*p)*wofz((q+1j*p)/np.sqrt(2))
    return np.where(z>=0,f,np.exp(-k*k/2)-f)


def interval_cf(lo,hi,k):
    # Reflect left-tail intervals; never pass infinities to wofz.
    flip=hi<=0
    l=np.where(flip,-hi,lo); h=np.where(flip,-lo,hi); k=np.where(flip,-k,k)
    a=np.where(np.isneginf(l),np.exp(-k*k/2),tail(np.where(np.isfinite(l),l,0.),k))
    b=np.where(np.isposinf(h),0.,tail(np.where(np.isfinite(h),h,0.),k))
    return np.where(lo<hi,a-b,0.)


def endpoint_boundary(d,v,w,a,b,k,order):
    """Price identity, starting at the nearest rank-one endpoint.

    Valid for nonsingular residual; intended only near singularity.
    Integrate angle theta from zero, r=sign*cos(theta).
    """
    q=np.hypot(v,w); h,j=(d/q).T
    A,B=np.linalg.solve(np.stack([v/q,w/q]),[a,b])
    A=A*k; B=B*k
    r=(v[0]*v[1]+w[0]*w[1])/(q[0]*q[1]); sign=1 if r>=0 else -1
    theta=np.arctan2(abs(v[0]*w[1]-v[1]*w[0])/(q[0]*q[1]),abs(r))
    h,j=h[:,None],j[:,None]; ab=A*B; delta=2*np.sin(theta/2)**2
    if sign==1: initial=tail(np.maximum(h,j),A+B)
    else: initial=interval_cf(h,-j,A-B)
    if np.any(sign*ab*delta>2):
        return None  # use direct residual integration instead
    out=np.exp(sign*ab*delta)*initial
    # Row-dependent angular split at the near-endpoint boundary layer.
    width=np.abs(j-sign*h)[:,0]
    edges=np.sort(np.clip(np.stack([np.zeros(len(h)),width/4,width,width*4,np.full(len(h),theta)],axis=1),0,theta),axis=1)
    x,wt=roots_legendre(order)
    for l,u in zip(edges[:,:-1].T,edges[:,1:].T):
        if not np.any(u>l): continue
        for xx,ww in zip(x,wt):
            th=(l+(u-l)*(1+xx)/2)[:,None]; wtrow=(u-l)[:,None]*ww/2
            th=np.where(wtrow>0,th,theta)
            sr=sign*np.cos(th); sr1=sign*2*np.sin(th/2)**2; sd=np.sin(th)
            eh=(h-sign*j+sr1*j)/sd; ej=(j-sign*h+sr1*h)/sd
            corner=np.exp(-.5*h*h-.5*ej*ej+1j*(A*h+B*j))/(2*np.pi)
            edgej=1j*A*sd/np.sqrt(2*np.pi)*np.exp(-.5*j*j+1j*(A*sr+B)*j)*tail(eh,A*sd)
            edgeh=1j*B*sd/np.sqrt(2*np.pi)*np.exp(-.5*h*h+1j*(A+B*sr)*h)*tail(ej,B*sd)
            out-=sign*wtrow*np.exp(sign*ab*(delta-2*np.sin(th/2)**2))*(corner+edgej+edgeh)
    return out


def boundary(t,d,v,w,a,b,k,order=8,pad=9.):
    """Integrate dominant residual analytically; only minor residual quadrature."""
    def limits(y):
        lo=np.full(y.shape,-np.inf); hi=np.full(y.shape,np.inf)
        ok=np.ones(y.shape,bool)
        for j in range(len(v)):
            z=d[:,j,None]-w[j]*y
            if v[j]>0: lo=np.maximum(lo,z/v[j])
            elif v[j]<0: hi=np.minimum(hi,z/v[j])
            else: ok &= z<0
        return lo,hi,ok
    if not np.any(w):
        lo,hi,ok=limits(np.zeros((len(t),1)))
        active=(lo[:,0]<hi[:,0]) & ok[:,0]
        out=np.zeros((len(t),len(k)),complex)
        out[active]=interval_cf(lo[active],hi[active],a*k)
        return out
    if len(v)==2 and np.linalg.norm(w)<.1*np.linalg.norm(v):
        result=endpoint_boundary(d,v,w,a,b,k,order)
        if result is not None: return result
    edges=[np.full(len(t),z) for z in (-pad,-4.,-2.,0.,2.,4.,pad) if abs(z)<=pad]
    edges.extend([np.full(len(t),-pad),np.full(len(t),pad)])
    if len(v)==2:
        den=w[0]*v[1]-w[1]*v[0]
        if den!=0: edges.append((d[:,0]*v[1]-d[:,1]*v[0])/den)
    for j in range(len(v)):
        if w[j]!=0 and abs(v[j])<.2*abs(w[j]):
            center=d[:,j]/w[j]; width=abs(v[j]/w[j])
            edges.extend(center+q*width for q in (-6,-2,0,2,6))
    edges=np.sort(np.clip(np.stack(edges,axis=1),-pad,pad),axis=1)
    x,wt=roots_legendre(order); half=np.diff(edges,axis=1)/2
    y=(edges[:,:-1,None]+half[:,:,None]*(1+x)).reshape(len(t),-1)
    weight=(half[:,:,None]*wt).reshape(len(t),-1)*np.exp(-y*y/2)/np.sqrt(2*np.pi)
    lo,hi,ok=limits(y)
    return np.einsum('tz,tzk->tk',weight*ok,interval_cf(lo[:,:,None],hi[:,:,None],a*k)*np.exp(1j*b*y[:,:,None]*k),optimize=True)


def field(mu,roots,beta,panels=96,nk=28,cutoff=14.,pad=9.,order=8,batch=8,return_info=False):
    """Target C-C beta beta' C, C=blockdiag(R R'). Roots explicit, no ridge.

    Roots have 1--3 rows and at most that many columns; nonproportional
    rows in multirow blocks required. Exact rank-two triples use 2 columns.
    """
    mu=np.asarray(mu,float); beta=np.asarray(beta,float); roots=tuple(np.asarray(R,float) for R in roots)
    if mu.ndim!=1 or beta.shape!=mu.shape or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(beta)) or sum(len(R) for R in roots)!=len(mu) or len(mu)<2: raise ValueError('invalid inputs')
    if any(R.ndim!=2 or not 1<=R.shape[1]<=R.shape[0]<=3 or not np.all(np.isfinite(R)) or np.linalg.matrix_rank(R)!=R.shape[1] for R in roots): raise ValueError('invalid roots')
    if any(not isinstance(q,(int,np.integer)) or q<1 for q in (panels,nk,order,batch)) or not np.all(np.isfinite([pad,cutoff])) or min(pad,cutoff)<=0: raise ValueError('invalid quadrature')
    scale=max(np.linalg.norm(R,axis=1).max() for R in roots); m=(mu-mu.min())/scale; beta=beta*scale
    specs=[]; knots=[]; s=[]; c=[]; vs=[]; off=0
    for R in roots:
        R=R/scale; b=beta[off:off+len(R)]; sd=np.linalg.norm(R,axis=1); h=R.T@b; cb=R@h
        if np.any(sd<=0): raise ValueError('zero marginal')
        s.extend(sd); c.extend(cb); vs.append(h@h); group=[]
        for i in range(len(R)):
            j=np.delete(np.arange(len(R)),i)
            if not len(j): group.append(None); continue
            Q=np.linalg.qr(R[i,:,None],mode='complete')[0]
            L=R[j]@Q[:,1:]; reg=(R[j]@R[i])/sd[i]**2
            U,S,V=np.linalg.svd(L,full_matrices=False)
            W=U*S; v=W[:,0] if len(S) else np.zeros(len(j)); w=W[:,1] if len(S)>1 else np.zeros(len(j))
            q=np.linalg.norm(L,axis=1)
            if np.any(q<=32*np.finfo(float).eps*sd[j]): raise ValueError('proportional local rows unsupported')
            hv=V@(Q[:,1:].T@h); a=hv[0]; z=hv[1] if len(hv)>1 else 0.
            A=1-reg; B=reg*m[off+i]-m[off+j]
            for aa,bb,qq in zip(A,B,q):
                if aa!=0: knots.append((-bb/aa,qq/abs(aa)))
            if len(j)==2:
                aa=A[0]*v[1]-A[1]*v[0]; bb=B[0]*v[1]-B[1]*v[0]; ww=w[0]*v[1]-w[1]*v[0]
                if aa!=0: knots.append((-bb/aa,abs(ww/aa)))
            group.append((A,B,v,w,a,z))
        specs.append((slice(off,off+len(R)),group)); off+=len(R)
    s,c,vs=np.array(s),np.array(c),np.array(vs); total=vs.sum()
    if total>1+16*np.finfo(float).eps: raise ValueError('observation variance exceeds one')
    adjust=np.sqrt(max(total,1.)); c/=adjust; vs/=adjust**2
    noise=max(0.,1-vs.sum())
    if noise<16*np.finfo(float).eps and np.count_nonzero(vs>0)<3: raise ValueError('singular observation needs three active blocks')
    lo=np.min(m-pad*s); hi=max(np.min(m[sl]+pad*s[sl]) for sl,_ in specs)
    step=(hi-lo)/panels; extra=[]
    for center,width in knots:
        extra.append(center)
        if 0<width<step: extra.extend(center+q*width for q in (-6,-2,2,6))
    extra=np.asarray(extra); edges=np.unique(np.r_[np.linspace(lo,hi,panels+1),extra[(extra>lo)&(extra<hi)]])
    dx=np.diff(edges); t=np.r_[(edges[:-1,None]+dx[:,None]*np.arange(4)/4).ravel(),edges[-1]]
    zz=(t[:,None]-m)/s; f=np.exp(-zz*zz/2)/(np.sqrt(2*np.pi)*s)
    k,kw=roots_legendre(nk); k=(k+1)*cutoff/2; kw=kw*cutoff/2*np.sqrt(2/np.pi)*np.exp(-noise*k*k/2)
    p=np.zeros(len(m)); half=np.repeat(dx/2,2)
    for start in range(0,nk,batch):
        slk=slice(start,start+batch); kk=k[slk]; g=f[:,None,:]*np.exp(1j*kk[None,:,None]*(zz*c/s)[:,None,:])
        for sl,group in specs:
            for i,spec in enumerate(group,sl.start):
                if spec is None: continue
                A,B,v,w,a,b=spec; d=t[:,None]*A+B
                g[:,:,i]*=boundary(t,d,v,w,a/adjust,b/adjust,kk,order,pad)
        G=np.stack([g[:,:,sl].sum(axis=2) for sl,_ in specs],axis=2)
        area=(G[:-2:2]+4*G[1::2]+G[2::2])*half[:,None,None]/6
        S=np.concatenate([np.cumsum(area[::-1],axis=0)[::-1],np.zeros((1,len(kk),len(specs)),complex)],axis=0)
        left=np.ones_like(S); right=np.ones_like(S)
        left[:,:,1:]=np.cumprod(S[:,:,:-1],axis=2); right[:,:,:-1]=np.cumprod(S[:,:,:0:-1],axis=2)[:,:,::-1]
        outer=g[::2].copy()
        for j,(sl,_) in enumerate(specs): outer[:,:,sl]*=(left[:,:,j]*right[:,:,j])[:,:,None]
        integ=np.sum((outer[:-2:2]+4*outer[1::2]+outer[2::2])*dx[:,None,None]/6,axis=0)
        p+=kw[slk]@integ.real
    info={'panels':len(dx),'mass_defect':abs(p.sum()-1)}
    return (p,info) if return_info else p
