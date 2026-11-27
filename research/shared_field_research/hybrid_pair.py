"""All-winner Fourier field for independent 1/2-person Gaussian blocks
conditioned on one common scalar Gaussian observation of value zero.
C_target=C_block-c c.T, c=C_block @ beta, beta.T @ c <= 1.
Research only; raw shares, no normalization or ridge.
"""
import numpy as np
from scipy.special import wofz, ndtr, roots_legendre, erfc


def truncated_cf(z,k):
    zp=np.abs(z); kp=np.where(z>=0,k,-k)
    tail=.5*np.exp(-.5*zp*zp+1j*kp*zp)*wofz((kp+1j*zp)/np.sqrt(2))
    return np.where(z>=0,tail,np.exp(-.5*k*k)-tail)


def prepare(mu,blocks,beta):
    mu,beta=np.asarray(mu,float),np.asarray(beta,float)
    if mu.ndim!=1 or beta.shape!=mu.shape or len(mu)<2 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(beta)):
        raise ValueError('invalid inputs')
    blocks=tuple(np.asarray(C,float) for C in blocks)
    if any(C.ndim!=2 or C.shape not in ((1,1),(2,2)) or not np.all(np.isfinite(C)) or not np.array_equal(C,C.T) or np.any(np.diag(C)<=0) for C in blocks) or sum(map(len,blocks))!=len(mu):
        raise ValueError('invalid blocks')
    scale=max(np.sqrt(np.diag(C)).max() for C in blocks)
    m=(mu-mu.min())/scale; beta=beta*scale
    specs=[]; knots=[]; s=[]; c=[]; vs=[]; off=0
    for C in blocks:
        C=C/scale**2; b=beta[off:off+len(C)]; sd=np.sqrt(np.diag(C)); R=C/np.outer(sd,sd)
        if len(C)==2 and abs(R[0,1])>=1: raise ValueError('local perfect correlation outside scope')
        cb=C@b; v=float(b@cb); c.extend(cb); vs.append(v); s.extend(sd)
        group=[]
        for i in range(len(C)):
            if len(C)==1: group.append(None); continue
            j=1-i; rho=R[i,j]; q=sd[j]*np.sqrt((1-rho)*(1+rho)); reg=C[j,i]/C[i,i]
            slope=(1-reg)/q; intercept=(reg*m[off+i]-m[off+j])/q
            if slope!=0: knots.append((-intercept/slope,1/abs(slope)))
            group.append((slope,intercept,b[j]*q))
        specs.append((slice(off,off+len(C)),group)); off+=len(C)
    s,c,vs=np.array(s),np.array(c),np.array(vs)
    total=vs.sum()
    if total>1+16*np.finfo(float).eps: raise ValueError('invalid observation variance')
    if total>1: # roundoff-only boundary repair
        c=c/np.sqrt(total); beta=beta/np.sqrt(total)
        specs=[(sl,[None if g is None else (g[0],g[1],g[2]/np.sqrt(total)) for g in group]) for sl,group in specs]
        vs=vs/total
    noise=max(0.,1-vs.sum())
    if noise<16*np.finfo(float).eps and np.count_nonzero(vs>0)<3: raise ValueError('singular observation requires three active blocks')
    # A singular global observation with >=3 independent active blocks
    # cannot make any two contestant contrasts deterministic.
    return m,s,c,vs,noise,specs,knots


def frequency_bound(vs,noise,K):
    """Absolute omitted-frequency bound per share, max over blocks."""
    r=np.maximum(1.,K*np.sqrt(vs)/np.sqrt(2/np.pi)); active=r>1
    q=active.sum()-active.astype(int); prod=np.log(r).sum()-np.log(r)
    alg=np.full(len(vs),np.inf); ok=q>=2
    alg[ok]=np.sqrt(2/np.pi)*K*np.exp(-prod[ok])/(q[ok]-1)
    gauss=erfc(K*np.sqrt(noise/2))/np.sqrt(noise) if noise>0 else np.inf
    return float(np.minimum(alg,gauss).max())


def field(mu,blocks,beta,panels=256,nk=48,cutoff=12.,pad=9.,batch=8,return_info=False):
    if any(not isinstance(x,(int,np.integer)) or x<1 for x in (panels,nk,batch)) or not np.isfinite(cutoff) or cutoff<=0 or not np.isfinite(pad) or pad<=0: raise ValueError('invalid quadrature')
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
                if spec is not None:
                    a,b,h=spec; g[:,:,i]*=truncated_cf((t*a+b)[:,None],kk[:,:,0]*h)
        G=np.stack([g[:,:,sl].sum(axis=2) for sl,_ in specs],axis=2)
        area=(G[:-2:2]+4*G[1::2]+G[2::2])*half[:,None,None]/6
        S=np.concatenate([np.cumsum(area[::-1],axis=0)[::-1],np.zeros((1,len(k[slk]),len(specs)),complex)],axis=0)
        left=np.ones_like(S); right=np.ones_like(S)
        left[:,:,1:]=np.cumprod(S[:,:,:-1],axis=2); right[:,:,:-1]=np.cumprod(S[:,:,:0:-1],axis=2)[:,:,::-1]
        outer=g[::2].copy()
        for j,(sl,_) in enumerate(specs): outer[:,:,sl]*=(left[:,:,j]*right[:,:,j])[:,:,None]
        integ=np.sum((outer[:-2:2]+4*outer[1::2]+outer[2::2])*dx[:,None,None]/6,axis=0)
        p+=w[slk]@integ.real
    info={'panels':len(dx),'mass_defect':abs(p.sum()-1),'frequency_tail':frequency_bound(vs,noise,cutoff),'performance_tail':(len(m)+1)*ndtr(-pad),'inner_survival_tail':np.sqrt(2/np.pi)*cutoff*(len(specs)-1)*ndtr(-pad)}
    return (p,info) if return_info else p
