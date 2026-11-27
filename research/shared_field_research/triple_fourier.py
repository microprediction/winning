"""Research: Fourier boundary field for Gaussian blocks of size 1--3.

Rotate the two residual Gaussian directions so one is the observation
score. The remaining inequalities form a real interval. A common grid
with an explicit kink-panel replacement is an experimental acceleration.
"""
import numpy as np
from scipy.special import ndtr, roots_legendre, wofz


def truncated_cf(z,k):
    zp=np.abs(z); kp=np.where(z>=0,k,-k)
    tail=.5*np.exp(-.5*zp*zp+1j*kp*zp)*wofz((kp+1j*zp)/np.sqrt(2))
    return np.where(z>=0,tail,np.exp(-.5*k*k)-tail)


def interval(lo,hi):
    return np.maximum(0.,np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo)))


def triple_boundary(t, thresholds, L, b, k, nz, zpad):
    """Row-dependent split quadrature, including deterministic constraints."""
    h=L.T@b; a=np.linalg.norm(h)
    direction=h/a if a else np.array([1.,0.])
    v=L@direction; w=L@np.array([-direction[1],direction[0]])
    nt=len(t); edges=[np.full(nt,-zpad),np.full(nt,zpad)]
    tiny=16*np.finfo(float).eps*np.linalg.norm(L)
    for j in range(2):
        if abs(v[j])>tiny:
            center=thresholds[:,j]/v[j]; width=abs(w[j]/v[j])
            edges.extend(center+q*width for q in (-6,-2,0,2,6))
    den=v[0]*w[1]-v[1]*w[0]
    if den!=0: edges.append((thresholds[:,0]*w[1]-thresholds[:,1]*w[0])/den)
    edges=np.sort(np.clip(np.stack(edges,axis=1),-zpad,zpad),axis=1)
    x,weight=roots_legendre(nz); widths=np.diff(edges,axis=1)/2
    z=(edges[:,:-1,None]+widths[:,:,None]*(1+x)).reshape(nt,-1)
    ww=(widths[:,:,None]*weight).reshape(nt,-1)
    lo=np.full_like(z,-np.inf); hi=np.full_like(z,np.inf); ok=np.ones_like(z,dtype=bool)
    for j in range(2):
        d=thresholds[:,j,None]-v[j]*z
        if w[j]>0: lo=np.maximum(lo,d/w[j])
        elif w[j]<0: hi=np.minimum(hi,d/w[j])
        else: ok &= d<0
    weights=ww*np.exp(-z*z/2)/np.sqrt(2*np.pi)*interval(lo,hi)*ok
    return np.einsum('tz,tzk->tk',weights,np.exp(1j*a*z[:,:,None]*k[None,None,:]),optimize=True)


def fast_boundary(t, thresholds, L, b, k, nz, zpad):
    """Shared observation-axis grid with row-specific kink-panel replacement."""
    h=L.T@b; a=np.linalg.norm(h); direction=h/a if a else np.array([1.,0.])
    v=L@direction; w=L@np.array([-direction[1],direction[0]])
    if np.any(abs(w)<.06*abs(v)) or np.any(w==0):
        return triple_boundary(t,thresholds,L,b,k,nz,zpad)
    x,wt=roots_legendre(4); cells=4*nz; step=2*zpad/cells
    edges=np.linspace(-zpad,zpad,cells+1); nodes=(edges[:-1,None]+step*(x+1)/2).ravel()
    def real(z):
        d=(thresholds[:,0,None]-v[0]*z)/w[0]; e=(thresholds[:,1,None]-v[1]*z)/w[1]
        if w[0]>0 and w[1]>0: mass=interval(np.maximum(d,e),np.inf)
        elif w[0]<0 and w[1]<0: mass=interval(-np.inf,np.minimum(d,e))
        elif w[0]>0: mass=interval(d,e)
        else: mass=interval(e,d)
        return mass*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    weights=real(nodes[None,:])*np.tile(wt*step/2,cells)
    out=weights@np.exp(1j*a*nodes[:,None]*k)
    den=v[0]*w[1]-v[1]*w[0]
    if den==0: return out
    kink=(thresholds[:,0]*w[1]-thresholds[:,1]*w[0])/den
    kink=np.clip(kink,-zpad,zpad); idx=np.minimum(cells-1,((kink+zpad)/step).astype(int)); idx=np.maximum(idx,0)
    l=edges[idx]; r=edges[idx+1]; row=np.arange(len(t))[:,None]
    oldidx=idx[:,None]*4+np.arange(4)
    def dot(z,weight):
        return np.einsum('tz,tzk->tk',weight,np.exp(1j*a*z[:,:,None]*k[None,None,:]),optimize=True)
    out-=dot(nodes[oldidx],weights[row,oldidx])
    for lo,hi in [(l,kink),(kink,r)]:
        width=(hi-lo)[:,None]/2; z=lo[:,None]+width*(1+x)
        out+=dot(z,real(z)*width*wt)
    return out


def transform(h,j,a,b,r,order=16):
    """Price differential identity; direction chosen to avoid exponential growth."""
    h,j,a,b=np.broadcast_arrays(h,j,a,b)
    ab=a*b; end=0. if np.all(ab*r>=0) else np.sign(r)
    if end==0:
        initial=truncated_cf(h,a)*truncated_cf(j,b)
    elif end==1:
        initial=truncated_cf(np.maximum(h,j),a+b)
    else:
        initial=np.where(h < -j,truncated_cf(h,a-b)-truncated_cf(-j,a-b),0.)
    out=np.exp(-ab*(r-end))*initial
    if r==end:return out
    x,w=roots_legendre(order); start,stop=np.arcsin(end),np.arcsin(r)
    theta=start+(stop-start)*(x+1)/2
    for th,wt in zip(theta,w*(stop-start)/2):
        s=np.sin(th); q=np.cos(th)
        density=np.exp(-.5*h*h-.5*((j-s*h)/q)**2+1j*(a*h+b*j))/(2*np.pi)
        termj=1j*a*q/np.sqrt(2*np.pi)*np.exp(-.5*j*j+1j*(a*s+b)*j)*truncated_cf((h-s*j)/q,a*q)
        termh=1j*b*q/np.sqrt(2*np.pi)*np.exp(-.5*h*h+1j*(a+b*s)*h)*truncated_cf((j-s*h)/q,b*q)
        out+=wt*np.exp(-ab*(r-s))*(density+termj+termh)
    return out


def price_boundary(t,thresholds,L,b,k,nz,zpad):
    q=np.sqrt(np.diag(L@L.T)); R=L@L.T/np.outer(q,q)
    return transform((thresholds[:,0]/q[0])[:,None],(thresholds[:,1]/q[1])[:,None],b[0]*q[0]*k[None,:],b[1]*q[1]*k[None,:],R[0,1],nz)


def prepare(mu,blocks,beta):
    mu,beta=np.asarray(mu,float),np.asarray(beta,float)
    if mu.ndim!=1 or beta.shape!=mu.shape or len(mu)<2 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(beta)): raise ValueError('invalid inputs')
    blocks=tuple(np.asarray(C,float) for C in blocks)
    if any(C.shape not in ((1,1),(2,2),(3,3)) or not np.all(np.isfinite(C)) or not np.array_equal(C,C.T) or np.any(np.diag(C)<=0) for C in blocks) or sum(map(len,blocks))!=len(mu): raise ValueError('invalid blocks')
    scale=max(np.sqrt(np.diag(C)).max() for C in blocks)
    m=(mu-mu.min())/scale; beta=beta*scale
    specs=[]; knots=[]; s=[]; c=[]; vs=[]; off=0
    for C in blocks:
        C=C/scale**2; b=beta[off:off+len(C)]; sd=np.sqrt(np.diag(C)); cb=C@b
        np.linalg.cholesky(C)
        v=float(b@cb); c.extend(cb); vs.append(v); s.extend(sd); group=[]
        for i in range(len(C)):
            j=np.delete(np.arange(len(C)),i)
            if not len(j): group.append(None); continue
            reg=C[j,i]/C[i,i]; T=C[np.ix_(j,j)]-np.outer(C[j,i],C[j,i])/C[i,i]; L=np.linalg.cholesky(T)
            a=1-reg; d=reg*m[off+i]-m[off+j]; q=np.sqrt(np.diag(T))
            for aa,dd,qq in zip(a,d,q):
                if aa!=0: knots.append((-dd/aa,qq/abs(aa)))
            if len(j)==2:
                for sign in (-1,1):
                    aa=a[0]/q[0]+sign*a[1]/q[1]; dd=d[0]/q[0]+sign*d[1]/q[1]
                    rr=T[0,1]/(q[0]*q[1]); width=np.sqrt(max(0,2*(1+sign*rr)))
                    if aa!=0: knots.append((-dd/aa,width/abs(aa)))
            group.append((a,d,L,b[j]))
        specs.append((slice(off,off+len(C)),group)); off+=len(C)
    s,c,vs=np.array(s),np.array(c),np.array(vs); total=vs.sum()
    if total>1+16*np.finfo(float).eps: raise ValueError('invalid observation variance')
    if total>1:
        c=c/np.sqrt(total); vs=vs/total
        specs=[(sl,[None if g is None else (g[0],g[1],g[2],g[3]/np.sqrt(total)) for g in group]) for sl,group in specs]
    noise=max(0.,1-vs.sum())
    if noise<16*np.finfo(float).eps and np.count_nonzero(vs>0)<3: raise ValueError('singular observation requires three active blocks')
    return m,s,c,vs,noise,specs,knots


def field(mu,blocks,beta,panels=128,nk=32,cutoff=14.,pad=9.,nz=12,zpad=9.,batch=8,return_info=False,method='rotation'):
    if any(not isinstance(x,(int,np.integer)) or x<1 for x in (panels,nk,nz,batch)) or min(cutoff,pad,zpad)<=0 or not np.all(np.isfinite([cutoff,pad,zpad])): raise ValueError('invalid quadrature')
    boundary={'rotation':triple_boundary,'fast':fast_boundary,'price':price_boundary}[method]
    m,s,c,vs,noise,specs,knots=prepare(mu,blocks,beta)
    lo=np.min(m-pad*s); hi=max(np.min(m[sl]+pad*s[sl]) for sl,_ in specs)
    step=(hi-lo)/panels; extra=[]
    for center,width in knots:
        extra.append(center)
        if width<2*step: extra.extend(center+q*width for q in (-8,-4,-2,-1,-.5,.5,1,2,4,8))
    extra=np.asarray(extra); edges=np.unique(np.r_[np.linspace(lo,hi,panels+1),extra[(extra>lo)&(extra<hi)]])
    dx=np.diff(edges); t=np.r_[(edges[:-1,None]+dx[:,None]*np.arange(4)/4).ravel(),edges[-1]]
    z=(t[:,None]-m)/s; f=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*s)
    k,w=roots_legendre(nk); k=(k+1)*cutoff/2; w=w*cutoff/2*np.sqrt(2/np.pi)*np.exp(-.5*noise*k*k)
    p=np.zeros(len(m)); half=np.repeat(dx/2,2)
    for start in range(0,nk,batch):
        slk=slice(start,start+batch); kk=k[slk][None,:,None]
        g=f[:,None,:]*np.exp(1j*kk*(z*c/s)[:,None,:])
        for sl,group in specs:
            for i,spec in enumerate(group,sl.start):
                if spec is None: continue
                a,b,L,h=spec; thresholds=t[:,None]*a+b
                if len(a)==1: g[:,:,i]*=truncated_cf(thresholds/L[0,0],kk[:,:,0]*h[0]*L[0,0])
                else: g[:,:,i]*=boundary(t,thresholds,L,h,k[slk],nz,zpad)
        G=np.stack([g[:,:,sl].sum(axis=2) for sl,_ in specs],axis=2)
        area=(G[:-2:2]+4*G[1::2]+G[2::2])*half[:,None,None]/6
        S=np.concatenate([np.cumsum(area[::-1],axis=0)[::-1],np.zeros((1,len(k[slk]),len(specs)),complex)],axis=0)
        left=np.ones_like(S); right=np.ones_like(S)
        left[:,:,1:]=np.cumprod(S[:,:,:-1],axis=2); right[:,:,:-1]=np.cumprod(S[:,:,:0:-1],axis=2)[:,:,::-1]
        outer=g[::2].copy()
        for j,(sl,_) in enumerate(specs): outer[:,:,sl]*=(left[:,:,j]*right[:,:,j])[:,:,None]
        integ=np.sum((outer[:-2:2]+4*outer[1::2]+outer[2::2])*dx[:,None,None]/6,axis=0)
        p+=w[slk]@integ.real
    info={'panels':len(dx),'mass_defect':abs(p.sum()-1)}
    return (p,info) if return_info else p
