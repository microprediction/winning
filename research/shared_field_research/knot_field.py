"""Research prototype: knot-resolved conditional shift convolution.

Explicit planar roots + local common shifts + scalar global observation.
No ridge or structural approximation. See iteration16_knot_field.md.
"""
import numpy as np
from scipy.special import roots_legendre, roots_hermitenorm, wofz
from numpy.polynomial.legendre import legvander, legint, legval


def interval(lo, hi, k):
    def tail(z,k):
        p=np.abs(z); q=np.where(z>=0,k,-k)
        f=.5*np.exp(-p*p/2+1j*q*p)*wofz((q+1j*p)/np.sqrt(2))
        return np.where(z>=0,f,np.exp(-k*k/2)-f)
    flip=hi<=0
    l=np.where(flip,-hi,lo); h=np.where(flip,-lo,hi); k=np.where(flip,-k,k)
    a=np.where(np.isneginf(l),np.exp(-k*k/2),tail(np.where(np.isfinite(l),l,0.),k))
    b=np.where(np.isposinf(h),0.,tail(np.where(np.isfinite(h),h,0.),k))
    return np.where(lo<hi,a-b,0.)


def phi(x): return np.exp(-x*x/2)/np.sqrt(2*np.pi)


def _spec(m,R,i,sigma,h,score):
    s=np.linalg.norm(R[i]); d=np.hypot(s,sigma)
    if s==0: raise ValueError('zero planar marginal')
    e=R[i]/s; q=np.array([-e[1],e[0]])
    j=np.delete(np.arange(len(R)),i); v=R[j]@q
    if np.any(abs(v)<=64*np.finfo(float).eps*np.linalg.norm(R[j],axis=1)):
        raise ValueError('proportional planar rows unsupported')
    a=(1-R[j]@e/s)/v; b=(m[i]-m[j])/v
    # bounds as functions of unshifted winning performance x: a*(x-m_i)+b
    cross=[]
    for k in range(len(a)):
        for l in range(k):
            if a[k]!=a[l]: cross.append(m[i]+(b[l]-b[k])/(a[k]-a[l]))
    A=e@h; B=q@h
    return (m[i],s,d,sigma,a,b,v>0,np.array(cross), (A*s+score*sigma)/d,(-A*sigma+score*s)/d,B)


def _bounds(x,sp):
    m,s,d,sigma,a,b,pos,*_=sp
    v=(x[:,None]-m)*a+b
    lo=v[:,pos].max(1) if np.any(pos) else np.full(len(x),-np.inf)
    hi=v[:,~pos].min(1) if np.any(~pos) else np.full(len(x),np.inf)
    return lo,hi


def _knots(specs,step,pad,left,right):
    knots=list(np.linspace(left,right,max(1,int(np.ceil((right-left)/step)))+1))
    for m,s,d,sigma,a,b,pos,cross,*_ in specs:
        # Conditional mean of unshifted x is m+s^2/d^2*(t-m).
        # Map x-knots to t; conditional width in x is sigma*s/d.
        slope=s*s/(d*d); width=sigma*d/s
        active=[(c,True) for c in cross]
        for A,B in zip(a,b):
            if abs(A)*step>1:
                active.extend((c,abs(A)*sigma*s/d>.25) for c in m+(np.array([-6.,-3.,0.,3.,6.])-B)/A)
        for c,sharp in active:
            t=m+(c-m)/slope
            if abs(t-m)>pad*d+pad*width: continue
            for z in ([0.] if sigma==0 or not sharp else [-6.,-2.,0.,2.,6.]):
                u=t+z*width
                if left<u<right: knots.append(u)
    return np.unique(knots)


def _nodes(knots,z):
    half=np.diff(knots)/2; mid=(knots[:-1]+knots[1:])/2
    return (mid[:,None]+half[:,None]*z).ravel(),half,mid


def _transfer(knots,half,mid,t,C,I):
    panel=np.clip(np.searchsorted(knots,t,side='right')-1,0,len(half)-1)
    x=np.clip((t-mid[panel])/half[panel],-1.,1.)
    # Density polynomial and its exact tail primitive on each panel.
    A=legvander(x,len(C)-1)@C
    J=legval(1.,I)[None,:]-legvander(x,len(C))@I
    return panel,A,J*half[panel,None]


def _prepare(t,sp,gh,gl,pad):
    m,s,d,sigma,a,b,pos,cross,A,B,H=sp
    z=(t-m)/d; f=phi(z)/d
    center=m+(s*s/d)*z; width=sigma*s/d
    if width==0:
        lo,hi=_bounds(center,sp); ok=lo<hi
        return np.flatnonzero(ok),lo[ok],hi[ok],A*z[ok],f[ok],H
    # Integrate the independent residual u. Bounds are affine in u;
    # split wherever the active lower/upper envelope can change.
    # Smooth rows use GH; steep or crossing rows use split Legendre.
    c=(center[:,None]-cross)/width
    steep=np.max(abs(a))*width>.25
    split=np.any(abs(c)<pad,axis=1)
    if steep:
        for aa,bb in zip(a,b):
            if abs(aa)*width>.25:
                x=m+(np.array([-6.,-3.,0.,3.,6.])-bb)/aa
                split |= np.any(abs((center[:,None]-x)/width)<pad,axis=1)
    live=abs(z)<pad
    row=np.flatnonzero(~split & live); u,w=gh
    rr=np.repeat(row,len(u)); uu=np.tile(u,len(row)); ww=np.tile(w,len(row))
    sr=np.flatnonzero(split & live)
    if len(sr):
        cuts=[np.broadcast_to(np.array([-pad,-pad/2,0.,pad/2,pad]),(len(sr),5)),np.clip(c[sr],-pad,pad)]
        if steep:
            # Resolve steep normal-CDF transitions as well as kinks.
            for aa,bb in zip(a,b):
                if abs(aa)*width>.25:
                    x=m+(np.array([-6.,-3.,0.,3.,6.])-bb)/aa
                    cuts.append(np.clip((center[sr,None]-x)/width,-pad,pad))
        cuts=np.sort(np.concatenate(cuts,axis=1),axis=1)
        left=cuts[:,:-1]; right=cuts[:,1:]; j,k=np.where(right>left)
        mid=(left[j,k]+right[j,k])/2; half=(right[j,k]-left[j,k])/2
        g,v=gl; q=mid[:,None]+half[:,None]*g
        rr=np.r_[rr,np.repeat(sr[j],len(g))]; uu=np.r_[uu,q.ravel()]; ww=np.r_[ww,(half[:,None]*v*phi(q)).ravel()]
    lo,hi=_bounds(center[rr]-width*uu,sp); ok=lo<hi
    rr=rr[ok]; uu=uu[ok]; ww=ww[ok]*f[rr]
    # sort once to make reduceat fast; no per-frequency scatter
    order=np.argsort(rr,kind='stable'); rr=rr[order]
    return rr,lo[ok][order],hi[ok][order],(A*z[rr]+B*uu[order]),ww[order],H


def field(mu,roots,shifts,beta,step=.5,order=6,inner=10,nk=32,cutoff=16.,pad=9.,batch=4,diagnostics=False):
    mu=np.asarray(mu,float); beta=np.asarray(beta,float); shifts=np.asarray(shifts,float); roots=[np.asarray(r,float) for r in roots]; n=len(mu)
    if mu.ndim!=1 or beta.shape!=mu.shape or shifts.shape!=(len(roots),) or not len(roots) or sum(map(len,roots))!=n or not np.all(np.isfinite(np.r_[mu,beta,shifts])) or np.any(shifts<0): raise ValueError('input')
    if any(r.ndim!=2 or r.shape[1]!=2 or len(r)<2 or not np.all(np.isfinite(r)) or np.linalg.matrix_rank(r)!=2 for r in roots): raise ValueError('planar root required')
    if not all(isinstance(v,(int,np.integer)) and v>0 for v in (order,inner,nk,batch)) or not np.all(np.isfinite([step,cutoff,pad])) or min(step,cutoff,pad)<=0: raise ValueError('quadrature')
    scale=max(np.sqrt(np.max(np.sum(r*r,axis=1))+s*s) for r,s in zip(roots,shifts)); mu=(mu-mu.min())/scale; beta=beta*scale; shifts=shifts/scale; roots=[r/scale for r in roots]
    slices=[]; hs=[]; scores=[]; off=0
    for R,s in zip(roots,shifts):
        sl=slice(off,off+len(R)); slices.append(sl); hs.append(R.T@beta[sl]); scores.append(s*beta[sl].sum()); off+=len(R)
    vs=np.array([h@h+a*a for h,a in zip(hs,scores)]); total=vs.sum(); eps=16*np.finfo(float).eps
    if total>1+eps: raise ValueError('observation variance exceeds one')
    if total>=1-eps and np.count_nonzero(vs)<3: raise ValueError('singular observation needs three active blocks')
    adjust=np.sqrt(max(1.,total)); noise=max(0.,1-total)
    specs=[_spec(mu[sl],R,i,s,h/adjust,a/adjust) for sl,R,s,h,a in zip(slices,roots,shifts,hs,scores) for i in range(len(R))]
    left=min(sp[0]-pad*sp[2] for sp in specs); right=max(sp[0]+pad*sp[2] for sp in specs)
    knots=[_knots([sp],step,pad,left,right) for sp in specs]
    z,w=roots_legendre(order); C=np.linalg.solve(legvander(z,order-1),np.eye(order)); I=legint(C,axis=0)
    t,half,_=_nodes(np.unique(np.concatenate(knots)),z); M=len(t)
    u,v=roots_hermitenorm(max(3,inner//2)); gh=(u,v/np.sqrt(2*np.pi)); gl=roots_legendre(inner)
    # The GH branch is for smooth rows, but its nodes can exceed pad;
    # if a kink lies outside pad the missed contribution is a normal tail.
    prepared=[]
    for sp,kn in zip(specs,knots):
        local,lh,lm=_nodes(kn,z)
        panel,A,J=_transfer(kn,lh,lm,t,C,I)
        rr,lo,hi,phase,wt,H=_prepare(local,sp,gh,gl,pad)
        start=np.flatnonzero(np.r_[True,rr[1:]!=rr[:-1]]) if len(rr) else np.empty(0,int)
        prepared.append((rr[start],start,lo,hi,phase,wt,H,len(local),lh,panel,A,J))
    if total==0: k=np.array([0.]); kw=np.array([1.]); nk=1
    else:
        k,kw=roots_legendre(nk); k=(k+1)*cutoff/2; kw=kw*cutoff/2*np.sqrt(2/np.pi)*np.exp(-noise*k*k/2)
    p=np.zeros(n); weights=(half[:,None]*w).ravel()
    for start in range(0,nk,batch):
        kk=k[start:start+batch]; g=np.zeros((M,len(kk),n),complex); S=[np.zeros((M,len(kk)),complex) for sl in slices]
        block=np.repeat(np.arange(len(slices)),[sl.stop-sl.start for sl in slices])
        for i,(row,idx,lo,hi,phase,wt,H,size,lh,panel,A,J) in enumerate(prepared):
            local=np.zeros((size,len(kk)),complex)
            if len(row):
                val=wt[:,None]*np.exp(1j*phase[:,None]*kk)*interval(lo[:,None],hi[:,None],H*kk)
                local[row]=np.add.reduceat(val,idx,axis=0)
            a=local.reshape(-1,order,len(kk)); area=lh[:,None]*np.einsum('q,pqk->pk',w,a)
            tail=np.cumsum(area[::-1],axis=0)[::-1]-area
            vals=a[panel]
            g[:,:,i]=np.einsum('mq,mqk->mk',A,vals)
            S[block[i]]+=tail[panel]+np.einsum('mq,mqk->mk',J,vals)
        R=np.ones_like(S[0]); suffix=[]
        for a in S[::-1]: suffix.append(R); R=R*a
        L=np.ones_like(R)
        for sl,a,r in zip(slices,S,suffix[::-1]):
            p[sl]+=kw[start:start+batch]@np.einsum('m,mki->ki',weights,(g[:,:,sl]*(L*r)[:,:,None]).real); L=L*a
    return (p,dict(nodes=M,local_nodes=sum(len(x[2]) for x in prepared),boundary_nodes=sum(x[7] for x in prepared))) if diagnostics else p
