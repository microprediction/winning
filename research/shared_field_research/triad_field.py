"""Positive real shared field for independent Gaussian blocks of size 1--3.

Each block has an arbitrary PSD covariance (positive marginal variances and
strictly positive pairwise conditional variances). Optional ordinary factors
can be conditioned on outside this kernel. No fit, ridge or normalization.
Research prototype; errors are absolute, not certified relative tail errors.
"""
import numpy as np
from scipy.special import ndtr, owens_t


def bvn(h, k, rho):
    """Bivariate standard normal lower orthant by Owen's T (absolute accuracy)."""
    h,k=np.broadcast_arrays(np.asarray(h,float),np.asarray(k,float))
    if rho == 1:
        return ndtr(np.minimum(h,k))
    if rho == -1:
        return np.maximum(0.,np.where(h> -k, np.where(h>0,ndtr(k)-ndtr(-h),ndtr(h)-ndtr(-k)),0.))
    if rho == 0:
        return ndtr(h)*ndtr(k)
    q=np.sqrt((1-rho)*(1+rho))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        a=(k/np.where(h==0,1.,h)-rho)/q
        b=(h/np.where(k==0,1.,k)-rho)/q
        p=.5*(ndtr(h)+ndtr(k))-owens_t(h,a)-owens_t(k,b)-.5*(np.signbit(h)!=np.signbit(k))
        p=np.where(h==0,.5*ndtr(k)+owens_t(k,rho/q),p)
        p=np.where(k==0,.5*ndtr(h)+owens_t(h,rho/q),p)
    upper=np.minimum(ndtr(h),ndtr(k))
    if np.any(~np.isfinite(p)) or np.any(p < -2e-14) or np.any(p>upper+2e-14):
        raise ArithmeticError('bivariate CDF outside probability bounds')
    # Roundoff correction only; no claim of relative accuracy in tiny tails.
    return np.clip(p,0.,upper)


def prepare(mu, blocks):
    mu=np.asarray(mu,float)
    if mu.ndim!=1 or len(mu)==0 or not np.all(np.isfinite(mu)):
        raise ValueError('invalid means')
    covs=[]; count=0; sd=[]
    for C in blocks:
        C=np.asarray(C,float)
        if C.ndim!=2 or C.shape[0]!=C.shape[1] or not 1<=len(C)<=3 or not np.all(np.isfinite(C)) or not np.array_equal(C,C.T) or np.any(np.diag(C)<=0):
            raise ValueError('invalid block')
        s=np.sqrt(np.diag(C)); R=C/s[:,None]/s[None,:]; R=(R+R.T)/2; np.fill_diagonal(R,1.)
        if np.linalg.eigvalsh(R)[0]<-32*np.finfo(float).eps:
            raise ValueError('non-PSD block')
        if np.any(np.abs(R[np.triu_indices(len(R),1)])>=1.):
            raise ValueError('perfectly correlated pair outside scope, including possible permanent ties')
        covs.append(R); sd.extend(s); count+=len(C)
    if count!=len(mu): raise ValueError('blocks must cover means in order')
    sd=np.array(sd); scale=sd.max(); m=(mu-mu.min())/scale; s=sd/scale
    specs=[]; knots=[]; sharp=[]; off=0
    for R in covs:
        end=off+len(R); ms=m[off:end]; ss=s[off:end]; group=[]
        for i in range(len(R)):
            j=np.delete(np.arange(len(R)),i); rr=R[j,i]
            q=np.sqrt((1-rr)*(1+rr)); a=(rr/ss[i]-1/ss[j])/q; b=(ms[j]/ss[j]-rr*ms[i]/ss[i])/q
            rho=0.
            if len(j)==2:
                raw=(R[j[0],j[1]]-rr[0]*rr[1])/(q[0]*q[1])
                if abs(raw)>1+1e-10: raise ValueError('invalid conditional correlation')
                rho=float(np.clip(raw,-1.,1.))
                sign=1. if rho>=0 else -1.
                den=a[0]-sign*a[1]
                if den!=0:
                    center=-(b[0]-sign*b[1])/den
                    knots.append(center)
                    width=np.sqrt(2*(1-abs(rho)))/abs(den)
                    if width>0: sharp.append((center,width))
            for ai,bi in zip(a,b):
                if ai!=0.:
                    knots.append(-bi/ai)
                    sharp.append((-bi/ai,1/abs(ai)))
            group.append((a,b,rho))
        specs.append((slice(off,end),group)); off=end
    return m,s,specs,np.asarray(knots),sharp


def field(mu, blocks, panels=256, pad=9., return_info=False):
    """All shares, with positive composite Simpson integration on a shared mesh.

    panels is the number of uniform base panels; insert conditional crossing
    points as extra panel boundaries. Refine panels and pad independently.
    Blocks must be contiguous, nonempty, size <=3. Returns raw (not normalized)
    shares. For singular triple blocks no perfect pair correlation is allowed.
    """
    if not isinstance(panels,(int,np.integer)) or panels<1 or not np.isfinite(pad) or pad<=0: raise ValueError('invalid mesh')
    m,s,specs,knots,sharp=prepare(mu,blocks)
    lo=np.min(m-pad*s); hi=max(np.min(m[sl]+pad*s[sl]) for sl,_ in specs)
    step=(hi-lo)/panels
    extra=[center+q*width for center,width in sharp if width<2*step for q in (-8,-4,-2,-1,-.5,.5,1,2,4,8)]
    knots=np.r_[knots,extra]
    edges=np.unique(np.r_[np.linspace(lo,hi,panels+1),knots[(knots>lo)&(knots<hi)]])
    # quarterpoints supply positive Simpson integrals on each half-panel;
    # these give survival at each outer Simpson endpoint and midpoint.
    dx=np.diff(edges); t=np.r_[(edges[:-1,None]+dx[:,None]*np.arange(4)/4).ravel(),edges[-1]]
    z=(t[:,None]-m)/s; g=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*s)
    for sl,group in specs:
        for i,(a,b,rho) in enumerate(group,sl.start):
            if len(a)==1: g[:,i]*=ndtr(t*a[0]+b[0])
            if len(a)==2: g[:,i]*=bvn(t*a[0]+b[0],t*a[1]+b[1],rho)
    g=g.reshape(len(t),len(m))
    G=np.stack([g[:,sl].sum(axis=1) for sl,_ in specs],axis=1)
    half_width=np.repeat(dx/2,2)
    area=(G[:-2:2]+4*G[1::2]+G[2::2])*half_width[:,None]/6
    S=np.vstack([np.cumsum(area[::-1],axis=0)[::-1],np.zeros((1,len(specs)))])
    # No normalization. A survival above 1 is a discretization diagnostic.
    left=np.ones_like(S); right=np.ones_like(S)
    left[:,1:]=np.cumprod(S[:,:-1],axis=1); right[:,:-1]=np.cumprod(S[:, :0:-1],axis=1)[:,::-1]
    outer=g[::2].copy()
    for j,(sl,_) in enumerate(specs): outer[:,sl]*=(left[:,j]*right[:,j])[:,None]
    p=np.sum((outer[:-2:2]+4*outer[1::2]+outer[2::2])*dx[:,None]/6,axis=0)
    info={'panels':len(dx),'block_mass_defect':float(np.max(abs(S[0]-1))), 'mass_defect':float(abs(p.sum()-1)), 'performance_tail_per_share':float((len(m)+len(specs)+1)*ndtr(-pad))}
    return (p,info) if return_info else p


def factor_field(mu, blocks, v, order=15, **kwargs):
    """Add one ordinary Gaussian factor vv' by outer Gauss-Hermite quadrature.

    Common loading is invisible to the race. Refine order independently of
    inner panels. Only shares are returned; diagnostics via inner field.
    """
    from scipy.special import roots_hermitenorm
    mu,v=np.asarray(mu,float),np.asarray(v,float)
    if v.shape!=mu.shape or not np.all(np.isfinite(v)) or not isinstance(order,(int,np.integer)) or order<1 or kwargs.get('return_info',False):
        raise ValueError('invalid factor or quadrature')
    blocks=tuple(blocks)
    v=v-v.mean()
    if not np.any(v):return field(mu,blocks,**kwargs)
    x,w=roots_hermitenorm(order)
    return sum(wi*field(mu+xi*v,blocks,**kwargs) for xi,wi in zip(x,w))/np.sqrt(2*np.pi)
