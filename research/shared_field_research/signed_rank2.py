"""Research: rank-two negative factors, polar Fourier shared field.
C = diag(sd) (I - A A.T) diag(sd), A shape (n,2).
No fitting, ridge, clipping or normalization. Cutoffs and quadrature separate.
"""
import numpy as np
from scipy.special import roots_legendre, ndtr
from signed_field import truncated_cf


def _validate(a):
    a=np.asarray(a,float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape[0]<2 or not np.all(np.isfinite(a)):
        raise ValueError('A must be finite (n,2)')
    noise=np.eye(2)-a.T@a
    ev,v=np.linalg.eigh(noise)
    tol=16*np.finfo(float).eps
    if ev.min() < -tol:
        raise ValueError('non-PSD downdate')
    if ev.min()<0:  # roundoff only; keep A and noise consistent
        a=a@((v*(1/np.sqrt(np.maximum(1.,1-ev))))@v.T)
        noise=np.eye(2)-a.T@a
        ev,v=np.linalg.eigh(noise)
    noise=(v*np.maximum(ev,0))@v.T
    return a,noise


def sector_bounds(a, cutoffs, pad=9.):
    """Real-arithmetic omitted-frequency bound for each sector and winner.

    Uniform sectors partition [0,pi]. Returns (m,n), already weighted by
    angular width/pi. Infinity means bound not established, not divergence.
    Add (n+1)*ndtr(-pad) to their sum for omitted performance domain.
    """
    a,noise=_validate(a)
    k=np.asarray(cutoffs,float)
    if k.ndim!=1 or len(k)==0 or np.any(~np.isfinite(k)) or np.any(k<=0):
        raise ValueError('positive finite sector cutoffs required')
    m=len(k);th=(np.arange(m)+.5)*np.pi/m
    u=np.stack([np.cos(th),np.sin(th)],axis=1)
    # For any direction in sector, ||u-u_center|| <= chord.
    chord=2*np.sin(np.pi/(4*m))
    b=np.maximum(0.,np.abs(u@a.T)-chord*np.linalg.norm(a,axis=1))
    r=np.maximum(1.,k[:,None]*b/np.sqrt(2/np.pi))
    active=r>1;q=active.sum(axis=1)[:,None]-active
    logprod=np.log(r).sum(axis=1)[:,None]-np.log(r)
    alg=np.full_like(r,np.inf);ok=q>2
    np.divide(k[:,None]**2*np.exp(-logprod),q-2,out=alg,where=ok)
    ev,v=np.linalg.eigh(noise);ev=np.where(ev>16*np.finfo(float).eps,ev,0.)
    sq=(v*np.sqrt(ev))@v.T
    # Directional Gaussian damping, including global minimum as floor.
    low=np.maximum(np.sqrt(ev.min()),np.linalg.norm(u@sq,axis=1)-chord*np.sqrt(ev.max()))**2
    gau=np.full(m,np.inf)
    np.divide(np.exp(-low*k*k/2),low,out=gau,where=low>0)
    return np.minimum(alg,gau[:,None])/m


def choose_cutoffs(a, sectors=16, tolerance=1e-8, cap=256.):
    """Dyadic cutoff per angular sector; refuses an uncertified domain.
    This controls Fourier truncation only, NOT quadrature or roundoff.
    """
    if sectors<1 or tolerance<=0 or cap<4:
        raise ValueError('invalid cutoff controls')
    k=np.full(sectors,4.)
    while True:
        bad=sector_bounds(a,k).max(axis=1)>tolerance/sectors
        if not np.any(bad):return k
        if np.any(k[bad]>=cap):raise ValueError('frequency tail not bounded at cap')
        k[bad]=np.minimum(2*k[bad],cap)


def field2(mu,sd,a,nt=96,nr=24,sectors=16,na=2,cutoffs=16.,pad=9.,batch=8):
    """Raw all-winner shares, polar Fourier quadrature.
    nt performance nodes; nr radial nodes per ray; na angular nodes per
    sector. Supply cutoffs or choose_cutoffs() separately. O(n*nt*nr*na*sectors).
    Supports singular noise if sector bounds establish absolute integrability;
    this is sufficient, not necessary. Does not support permanent ties.
    """
    a,noise=_validate(a);mu=np.asarray(mu,float);sd=np.asarray(sd,float)
    n=len(a)
    if mu.shape!=(n,) or sd.shape!=(n,) or np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sd)) or np.any(sd<=0):
        raise ValueError('invalid means or scales')
    if min(nt,nr,sectors,na,batch)<1 or pad<=0:raise ValueError('invalid quadrature')
    k=np.broadcast_to(np.asarray(cutoffs,float),(sectors,))
    bounds=sector_bounds(a,k)
    if np.linalg.eigvalsh(noise).min()<=16*np.finfo(float).eps:
        # Establish integrability (some b_j may be zero). Reject any pair
        # with effectively zero contrast variance rather than change tie law.
        if not np.all(np.isfinite(bounds)):
            raise ValueError('singular constraint without established integrability')
        b=sd[:,None]*a
        leverage=np.sum(a*a,axis=1)
        if np.sort(leverage)[-2:].sum()>=1-16*np.finfo(float).eps:
            for i in range(n):
                s=(sd[i]**2+sd[i+1:]**2)
                if np.any(s-np.sum((b[i]-b[i+1:])**2,axis=1)<=16*np.finfo(float).eps*s):
                    raise ValueError('possible permanent tie')
    scale=sd.max();mu=(mu-mu.min())/scale;sd=sd/scale
    lo=np.min(mu-pad*sd);hi=np.min(mu+pad*sd)
    tx,tw=roots_legendre(nt);t=lo+(tx+1)*(hi-lo)/2;tw*= (hi-lo)/2
    ax,aw=roots_legendre(na);rx,rw=roots_legendre(nr)
    theta=(np.arange(sectors)[:,None]+(ax+1)/2)*np.pi/sectors
    u=np.stack([np.cos(theta),np.sin(theta)],axis=-1)
    r=k[:,None]*(rx+1)/2
    f=(u[:,:,None,:]*r[:,None,:,None]).reshape(-1,2)
    w=np.broadcast_to(aw[None,:,None]*rw[None,None,:]*k[:,None,None]*r[:,None,:]/(4*sectors),(sectors,na,nr)).ravel().copy()
    w*=np.exp(-.5*np.einsum('bi,ij,bj->b',f,noise,f))
    ka=(f@a.T)[None,:,:];p=np.zeros(n)
    for start in range(0,nt,batch):
        sl=slice(start,start+batch);z=(t[sl,None]-mu)/sd
        pdf=np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*sd);z=z[:,None,:]
        c=truncated_cf(z,ka)
        l=np.ones_like(c);r=np.ones_like(c)
        l[:,:,1:]=np.cumprod(c[:,:,:-1],axis=2)
        r[:,:,:-1]=np.cumprod(c[:,:,:0:-1],axis=2)[:,:,::-1]
        win=l*r*np.exp(1j*ka*z)*pdf[:,None,:]
        p+=np.einsum('t,k,tki->i',tw[sl],w,win.real)
    return p
