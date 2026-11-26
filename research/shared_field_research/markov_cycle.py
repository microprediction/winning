"""Research: all Gaussian min-win shares for a cyclic tridiagonal precision.

The cycle is contracted exactly at fixed quadrature, using a boundary
insertion in each vertex's shared transfer field. No loopy BP.
"""
import numpy as np
from scipy.special import roots_legendre


def cycle_marginals(diag, edge):
    """O(n) marginal standard deviations and logdet of cycle precision.

    edge[i] is Q[i,(i+1)%n]. Eliminate the open path, then its last
    separator. Inputs must be finite; raises on nonpositive pivots.
    """
    d, e = np.asarray(diag, float), np.asarray(edge, float)
    n = d.size
    if (n < 3 or d.shape != (n,) or e.shape != (n,) or
            not np.all(np.isfinite(d)) or not np.all(np.isfinite(e))):
        raise ValueError('finite diagonal and cycle edges, n >= 3 required')
    p = d[:-1].copy()
    l = np.empty(n-2)
    b = np.zeros(n-1)
    b[0], b[-1] = e[-1], e[-2]
    for i in range(n-2):
        if p[i] <= 0:
            raise ValueError('precision must be positive definite')
        l[i] = e[i]/p[i]
        p[i+1] -= l[i]*e[i]
        b[i+1] -= l[i]*b[i]
    if p[-1] <= 0:
        raise ValueError('precision must be positive definite')
    s = d[-1] - np.dot(b, b/p)
    if not np.isfinite(s) or s <= 0:
        raise ValueError('precision must be positive definite')
    v = b/p
    invdiag = 1/p
    for i in range(n-3, -1, -1):
        v[i] -= l[i]*v[i+1]
        invdiag[i] += l[i]**2*invdiag[i+1]
    var = np.r_[invdiag+v*v/s, 1/s]
    return np.sqrt(var), np.log(p).sum()+np.log(s)


def cycle_shares(mu, diag, edge, *, L=64, M=32, span=9.):
    """Raw shares; increasing L,M then span converges for fixed SPD Q.

    Q is represented by its diagonal and its cyclic off-diagonal. Very
    sharp or poorly conditioned precision can need many grid points.
    Matrix products are scaled, but cancellation of extreme factor
    exponents is not guaranteed; check resolution and mass diagnostics.
    """
    mu, d, e = map(lambda x: np.asarray(x, float), (mu, diag, edge))
    n = d.size
    if mu.shape != (n,) or not np.all(np.isfinite(mu)):
        raise ValueError('finite mean vector required')
    if (any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or k < 2 for k in (L, M)) or not np.isfinite(span) or span <= 0):
        raise ValueError('L,M integers >= 2 and positive finite span required')
    sd, logdet = cycle_marginals(d, e)
    # Transform to standardized coordinates; precision scale is irrelevant.
    qd = d*sd*sd
    qe = e*sd*np.roll(sd,-1)
    lognorm = .5*(logdet-n*np.log(2*np.pi))+np.log(sd).sum()
    scale = sd.max()
    mu, sd = (mu-mu[0])/scale, sd/scale
    if (not np.all(np.isfinite(qd)) or not np.all(np.isfinite(qe)) or
            not np.all(np.isfinite(mu)) or not np.all(sd > 0)):
        raise ValueError('utility range exceeds prototype arithmetic')
    a, b = np.min(mu-span*sd), np.min(mu+span*sd)
    if not np.isfinite(a+b) or b <= a:
        raise ValueError('invalid threshold window')
    x, wx = roots_legendre(L)
    x, wx = a+(b-a)/2*(1+x), wx*(b-a)/2
    cut = (x[:,None]-mu)/sd
    lo = np.clip(cut,-span,span)
    q,w = roots_legendre(M)
    width = (span-lo)/2
    z = np.concatenate([lo[:,:,None]+width[:,:,None]*(1+q),lo[:,:,None]],axis=2)
    w = np.concatenate([width[:,:,None]*w,np.zeros((L,n,1))],axis=2)
    # Each pair potential holds half of each incident unary exponent.
    kernels, shifts = [], []
    for i in range(n):
        j = (i+1)%n
        u, v = z[:,i,:,None], z[:,j,None,:]
        h = -.25*qd[i]*u*u-qe[i]*u*v-.25*qd[j]*v*v
        hmax = h.max(axis=(1,2))
        kernels.append(np.exp(h-hmax[:,None,None]))
        shifts.append(hmax)
    A = [w[:,i,:,None]*kernels[i] for i in range(n)]
    eye = np.broadcast_to(np.eye(M+1),(L,M+1,M+1))

    def scaled(a):
        m = a.max(axis=(1,2))
        # A zero matrix contributes zero, not NaN.
        safe = np.where(m > 0,m,1.)
        return a/safe[:,None,None], np.log(safe)

    P, lp = [eye], [np.zeros(L)]
    for i in range(n-1):
        p,t = scaled(P[-1]@A[i])
        P.append(p); lp.append(lp[-1]+t)
    S, ls = eye, np.zeros(L)
    out = np.empty(n)
    common = lognorm+np.sum(shifts,axis=0)
    for i in range(n-1,-1,-1):
        # d trace(A0 ... A[n-1])/d w_i(boundary), evaluated at zero.
        v = (S@P[i][:,:,-1,None])[:,:,0]
        val = np.sum(kernels[i][:,-1,:]*v,axis=1)
        mask = (np.abs(cut[:,i]) <= span)&(val > 0)
        # Combine the common scales in log space before exponentiation.
        out[i] = np.sum(wx[mask]*np.exp(common[mask]+ls[mask]+lp[i][mask]
                                        +np.log(val[mask])-np.log(sd[i])))
        if i:
            S,t = scaled(A[i]@S)
            ls += t
    if not np.all(np.isfinite(out)):
        raise FloatingPointError('nonfinite quadrature result')
    return out


def dense_precision(diag, edge):
    """Dense verification matrix only, not used by the pricing engine."""
    Q = np.diag(diag)
    i = np.arange(len(diag)); j = (i+1)%len(diag)
    Q[i,j] = Q[j,i] = edge
    return Q
