"""Research: planar contrast blocks plus independent local common shifts.
Scalar global observation; no ridge. Uniform periodic FFT convolution or knot-split spectral coefficients.
"""
import numpy as np
from scipy.special import roots_legendre, wofz
from scipy.fft import fft, ifft, fftfreq


def interval(lo,hi,k):
    def tail(z,k):
        p=np.abs(z); q=np.where(z>=0,k,-k)
        f=.5*np.exp(-p*p/2+1j*q*p)*wofz((q+1j*p)/np.sqrt(2))
        return np.where(z>=0,f,np.exp(-k*k/2)-f)
    flip=hi<=0
    l=np.where(flip,-hi,lo); h=np.where(flip,-lo,hi); k=np.where(flip,-k,k)
    a=np.where(np.isneginf(l),np.exp(-k*k/2),tail(np.where(np.isfinite(l),l,0.),k))
    b=np.where(np.isposinf(h),0.,tail(np.where(np.isfinite(h),h,0.),k))
    return np.where(lo<hi,a-b,0.)


def decompose(R):
    """Rotate explicit root into contrast span and independent common shift.
    Require two contrast directions and at most one extra root column.
    No positive singular direction is truncated.
    """
    R=np.asarray(R,float)
    if R.ndim!=2 or R.shape[1] not in (2,3) or len(R)<3: raise ValueError('root shape')
    if R.shape[1]==3 and len(R)>3:
        raise ValueError('use explicit planar and common-shift roots for larger blocks')
    if R.shape[1]==2: return R,0.
    _,s,v=np.linalg.svd(R[1:]-R[0],full_matrices=True)
    if s[-1]<=64*np.finfo(float).eps*s[0]: raise ValueError('unresolved contrast rank')
    a=R@v[2]; return R@v[:2].T,abs(a.mean())


def shifted(g,omega,k,sigma,h):
    if sigma==0: return g*np.exp(-.5*(k*h)**2)
    return ifft(fft(g,axis=0)*np.exp(-.5*(sigma*omega[:,None]-k[None,:]*h)**2),axis=0)


def survival(g,dx):
    # append a zero density at the periodic right endpoint (tail negligible).
    G=np.concatenate([g,np.zeros_like(g[:1])],axis=0)
    area=dx/3*(G[:-2:2]+4*G[1::2]+G[2::2])
    S=np.empty_like(G); S[-1]=0
    S[:-1:2]=np.cumsum(area[::-1],axis=0)[::-1]
    S[1::2]=S[:-1:2]-dx/12*(5*G[:-2:2]+8*G[1::2]-G[2::2])
    return S[:-1]


def spectral_spec(mm,R,i,e,q,s,H,adjust,omega,sigma,score,cutoff,left,dx,pad,local_order):
    # Integrate the Fourier coefficient of the *unshifted* boundary directly
    # on a knot-split quadrature; only the smoothed active band is needed.
    j=np.delete(np.arange(len(R)),i); v=R[j]@q
    a=(s-R[j]@e)/v; b=(mm[i]-mm[j])/v
    limit=(pad+cutoff*abs(score))/sigma
    mask=np.abs(omega)<=limit; w=omega[mask]
    # in standardized winner z coordinates; crossing knots include all
    # pairs in this prototype, fine for triples.
    knots=list(np.linspace(-pad,pad,max(32,int(2*pad*s*limit/8))+1))
    for A,B in zip(a,b):
        if A!=0: knots.extend((z-B)/A for z in (-8,-3,0,3,8))
    for j in range(len(a)):
        for k in range(j):
            if a[j]!=a[k]: knots.append((b[k]-b[j])/(a[j]-a[k]))
    knots=np.unique([v for v in knots if -pad<=v<=pad]); nodes,weights=roots_legendre(local_order)
    z=(knots[:-1,None]+knots[1:,None])/2+np.diff(knots)[:,None]*nodes/2
    wt=(np.diff(knots)[:,None]*weights/2).ravel(); z=z.ravel()
    lo=np.max((z[:,None]*a+b)[:,v>0],axis=1) if np.any(v>0) else np.full(len(z),-np.inf)
    hi=np.min((z[:,None]*a+b)[:,v<0],axis=1) if np.any(v<0) else np.full(len(z),np.inf)
    ok=lo<hi; z=z[ok]; lo=lo[ok]; hi=hi[ok]; wt=wt[ok]*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    E=np.exp(-1j*w[:,None]*(mm[i]+s*z[None,:]-left))*wt[None,:]/dx
    return (mask,w,z,lo,hi,E,e@H/adjust,q@H/adjust)


def field(mu,roots,shifts,beta,M=2048,nk=32,cutoff=16.,pad=10.,batch=8,spectral=True,local_order=8):
    mu=np.asarray(mu,float); beta=np.asarray(beta,float); shifts=np.asarray(shifts,float); roots=tuple(np.asarray(r,float) for r in roots)
    n=len(mu); B=len(roots)
    if mu.ndim!=1 or beta.shape!=mu.shape or shifts.shape!=(B,) or not B or sum(len(r) for r in roots)!=n or not np.all(np.isfinite(np.r_[mu,beta,shifts])) or np.any(shifts<0): raise ValueError('input')
    if any(r.ndim!=2 or r.shape[1]!=2 or len(r)<2 or not np.all(np.isfinite(r)) or np.linalg.matrix_rank(r)!=2 for r in roots): raise ValueError('planar root required')
    if not all(isinstance(v,(int,np.integer)) and v>0 for v in (M,nk,batch,local_order)) or M<4 or M%2 or not np.all(np.isfinite([cutoff,pad])) or min(cutoff,pad)<=0: raise ValueError('quadrature')
    scale=max(np.sqrt(np.max(np.sum(r*r,axis=1))+s*s) for r,s in zip(roots,shifts))
    m=(mu-mu.min())/scale; beta=beta*scale; shifts=shifts/scale; roots=[r/scale for r in roots]
    slices=[]; h=[]; hs=[]; off=0; sd=[]
    for R,s in zip(roots,shifts):
        sl=slice(off,off+len(R)); slices.append(sl); h.append(R.T@beta[sl]); hs.append(s*beta[sl].sum()); sd.extend(np.sqrt(np.sum(R*R,axis=1)+s*s)); off+=len(R)
    vs=np.array([a@a+b*b for a,b in zip(h,hs)]); total=vs.sum()
    if total>1+16*np.finfo(float).eps: raise ValueError('observation variance exceeds one')
    if total>=1-16*np.finfo(float).eps and np.count_nonzero(vs)>0 and np.count_nonzero(vs)<3: raise ValueError('singular observation needs three active blocks')
    adjust=np.sqrt(max(1.,total)); noise=max(0.,1-total)
    sd=np.array(sd); left=np.min(m-pad*sd); right=np.max(m+pad*sd); dx=(right-left)/M; t=left+dx*np.arange(M); omega=2*np.pi*fftfreq(M,dx)
    specs=[]
    for R,sl,H,sigma,score in zip(roots,slices,h,shifts,hs):
        group=[]; mm=m[sl]
        for i,r in enumerate(R):
            s=np.linalg.norm(r)
            if s==0: raise ValueError('zero planar marginal')
            e=r/s; q=np.array([-e[1],e[0]]); j=np.delete(np.arange(len(R)),i); v=R[j]@q; reg=R[j]@e/s
            if np.any(abs(v)<=64*np.finfo(float).eps*np.linalg.norm(R[j],axis=1)): raise ValueError('proportional planar rows unsupported')
            z=(t-mm[i])/s; d=(t[:,None]-mm[j])-z[:,None]*(R[j]@e)
            bound=d/v; pos=v>0
            l=np.max(bound[:,pos],axis=1) if np.any(pos) else np.full(M,-np.inf)
            u=np.min(bound[:,~pos],axis=1) if np.any(~pos) else np.full(M,np.inf)
            spec=None
            if spectral and sigma>=.05:
                spec=spectral_spec(mm,R,i,e,q,s,H,adjust,omega,sigma,score/adjust,cutoff,left,dx,pad,local_order)
            group.append((z,l,u,np.exp(-z*z/2)/(np.sqrt(2*np.pi)*s),e@H/adjust,q@H/adjust,spec))
        specs.append(group)
    if total==0: k=np.array([0.]); kw=np.array([1.]); nk=1
    else:
        k,kw=roots_legendre(nk); k=(k+1)*cutoff/2; kw=kw*cutoff/2*np.sqrt(2/np.pi)*np.exp(-noise*k*k/2)
    p=np.zeros(n); weight=np.ones(M)*2; weight[1::2]=4; weight[0]=1
    for start in range(0,nk,batch):
        kk=k[start:start+batch]; g=np.zeros((M,len(kk),n),complex); S=[]
        for sl,group,s,H in zip(slices,specs,shifts,hs):
            for i,(z,l,u,f,a,b,spec) in enumerate(group):
                if spec is not None:
                    mask,w,zz,ll,uu,E,aa,bb=spec
                    values=np.exp(1j*zz[:,None]*aa*kk)*interval(ll[:,None],uu[:,None],bb*kk)
                    coef=np.zeros((M,len(kk)),complex)
                    coef[mask]=(E@values)*np.exp(-.5*(s*w[:,None]-kk[None,:]*H/adjust)**2)
                    g[:,:,sl.start+i]=ifft(coef,axis=0)
                    continue
                ok=l<u; raw=np.zeros((M,len(kk)),complex)
                raw[ok]=f[ok,None]*np.exp(1j*z[ok,None]*a*kk)*interval(l[ok,None],u[ok,None],b*kk)
                g[:,:,sl.start+i]=shifted(raw,omega,kk,s,H/adjust)
            S.append(survival(g[:,:,sl].sum(axis=2),dx))
        # prefix and suffix, no division.
        R=np.ones_like(S[0]); suffix=[]
        for a in S[::-1]: suffix.append(R); R=R*a
        L=np.ones_like(R)
        for sl,a,r in zip(slices,S,suffix[::-1]):
            vals=g[:,:,sl]*(L*r)[:,:,None]
            p[sl]+=kw[start:start+batch]@np.einsum('m,mki->ki',weight,vals.real)*(dx/3)
            L=L*a
    return p
