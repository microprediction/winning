"""Research: all min-win shares on a bounded-width Gaussian precision graph."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import roots_legendre


def prepare(Q, order=None, max_width=3):
    """Sparse Gaussian elimination in the supplied order; no dropped fill.

    Returns conditional regression coefficients and selected marginal variances.
    Width is structural (including exact-zero numerical fill). No ridge.
    """
    Q = csr_matrix(Q, dtype=float, copy=True); Q.sum_duplicates()
    n = Q.shape[0]
    if n == 0 or Q.shape != (n, n) or not np.isfinite(Q.data).all():
        raise ValueError('finite square nonempty precision required')
    if (Q-Q.T).nnz:
        raise ValueError('exactly symmetric precision required')
    order = np.arange(n) if order is None else np.asarray(order)
    if (order.shape != (n,) or order.dtype.kind not in 'iu' or
            not np.array_equal(np.sort(order), np.arange(n))):
        raise ValueError('order must be a permutation')
    Q = Q[order][:, order].tocsr(); Q.eliminate_zeros()
    rows = [dict(zip(Q.indices[Q.indptr[i]:Q.indptr[i+1]],
                     Q.data[Q.indptr[i]:Q.indptr[i+1]])) for i in range(n)]
    sep, beta, pivot = [], [], []
    for i in range(n):
        row = rows[i]; p = row.get(i, 0.)
        if not np.isfinite(p) or p <= 0:
            raise ValueError('precision must be positive definite')
        s = tuple(sorted(j for j in row if j > i))
        if len(s) > max_width:
            raise ValueError('elimination width exceeds max_width')
        a = np.array([row[j] for j in s]); b = -a/p
        sep.append(s); beta.append(b); pivot.append(p)
        for k,j in enumerate(s):
            rows[j].pop(i, None)
            for l in range(k, len(s)):
                h = s[l]; v = rows[j].get(h, 0.) + a[k]*b[l]
                rows[j][h] = rows[h][j] = v
    # Selected inverse: each later-neighbor set is a clique in the fill graph.
    cov = [{} for _ in range(n)]
    for i in range(n-1,-1,-1):
        s,b,p = sep[i],beta[i],pivot[i]
        C = np.array([[cov[min(j,k)][max(j,k)] for k in s] for j in s]).reshape(len(s),len(s))
        cb = C@b
        cov[i].update(zip(s, cb)); cov[i][i] = 1/p + b@cb
    sd = np.sqrt([cov[i][i] for i in range(n)])
    if not np.all(np.isfinite(sd) & (sd > 0)):
        raise FloatingPointError('invalid marginal variance')
    return order, sep, beta, np.asarray(pivot), sd


def _align(a, small, big):
    # All variable scopes are in elimination order; axis 0 is threshold batch.
    return a.reshape((a.shape[0],)+tuple(a.shape[1+small.index(j)] if j in small else 1 for j in big))


def width_shares(mu, Q, order=None, *, L=64, M=24, span=9., max_width=3, batch=4):
    """Raw probabilities; SPD precision only. Nested positive quadrature.

    Complexity O(n L M**(width+1)) for fixed width; sparse preparation
    O(n width**2), memory O(n batch M**(width+1)). No covariance fit.
    """
    order, sep, beta, pivot, sd = prepare(Q, order, max_width)
    n = len(order); mu = np.asarray(mu, float)
    if mu.shape != (n,) or not np.isfinite(mu).all():
        raise ValueError('finite mean vector required')
    if any(not isinstance(k, (int,np.integer)) or k < 1 for k in (L,M,batch)) or not np.isfinite(span) or span <= 0:
        raise ValueError('positive integer resolutions and positive finite span required')
    mu = mu[order]; scale = max(sd); mu = (mu-mu[0])/scale
    if not np.isfinite(mu).all():
        raise ValueError("utility range exceeds prototype arithmetic")
    # Standardized Gaussian conditional: z_i | z_s ~ N(b_i z_s, tau_i**2).
    tau = 1/np.sqrt(pivot)/sd
    beta = [b*sd[list(s)]/sd[i] for i,(s,b) in enumerate(zip(sep,beta))]
    sd = sd/scale
    scopes = [(i,)+s for i,s in enumerate(sep)]
    children = [[] for _ in range(n)]; roots = []
    for i,s in enumerate(sep):
        (children[s[0]] if s else roots).append(i)
    a,b = np.min(mu-span*sd), np.min(mu+span*sd)
    x,wx = roots_legendre(L); x,wx = a+(b-a)*(1+x)/2, wx*(b-a)/2
    q,w = roots_legendre(M); out = np.zeros(n)
    for start in range(0,L,batch):
        sl = slice(start,start+batch); cut = (x[sl,None]-mu)/sd
        lo = np.clip(cut,-span,span); half = (span-lo)/2
        z = np.concatenate((lo[:,:,None]+half[:,:,None]*(1+q),lo[:,:,None]),axis=2)
        weights = np.concatenate((half[:,:,None]*w,np.zeros((*lo.shape,1))),axis=2)
        K,H,msg,scales = [],[],[],[]
        def norm(v):
            m = v.reshape(len(cut),-1).max(axis=1)
            safe = np.where(m > 0,m,1.)
            return v/safe.reshape((-1,)+(1,)*(v.ndim-1)),np.log(safe)
        for i,scope in enumerate(scopes):
            resid = _align(z[:,i],(i,),scope)
            for j,coef in zip(sep[i],beta[i]):
                resid = resid-coef*_align(z[:,j],(j,),scope)
            k = np.exp(-.5*(resid/tau[i])**2)/(tau[i]*np.sqrt(2*np.pi))
            K.append(k); h=k
            for j in children[i]:
                h=h*_align(msg[j],sep[j],scope)
            H.append(h)
            m,ls=norm(np.sum(h*_align(weights[:,i],(i,),scope),axis=1))
            msg.append(m); scales.append(ls+sum((scales[j] for j in children[i]),np.zeros(len(cut))))
        ctx,cs=[None]*n,[None]*n
        # Root messages are scalars. Prefix/suffix products handle exact zeros.
        pref=[np.ones(len(cut))]; ps=[np.zeros(len(cut))]
        for i in roots:
            pref.append(pref[-1]*msg[i]); ps.append(ps[-1]+scales[i])
        suffix=np.ones(len(cut)); ss=np.zeros(len(cut))
        for k in range(len(roots)-1,-1,-1):
            i=roots[k]; ctx[i]=pref[k]*suffix; cs[i]=ps[k]+ss
            suffix=suffix*msg[i]; ss=ss+scales[i]
        for i in range(n-1,-1,-1):
            scope=scopes[i]; outside=_align(ctx[i],sep[i],scope)
            subscale=sum((scales[j] for j in children[i]),np.zeros(len(cut)))
            val=(H[i][:,-1]*outside[:,0]).reshape(len(cut),-1).sum(axis=1)
            ok=(np.abs(cut[:,i])<=span)&(val>0)
            out[i]+=np.sum(wx[sl][ok]*np.exp(np.log(val[ok])+cs[i][ok]+subscale[ok]-np.log(sd[i])))
            # Reverse derivative: exclude child's message using prefix/suffix
            # products, so boundary zero weights and underflowed zeros are safe.
            kids=children[i]
            pref=[outside*K[i]*_align(weights[:,i],(i,),scope)]
            ps=[cs[i]]
            for j in kids:
                pref.append(pref[-1]*_align(msg[j],sep[j],scope)); ps.append(ps[-1]+scales[j])
            suffix=1.; ss=np.zeros(len(cut))
            for k in range(len(kids)-1,-1,-1):
                j=kids[k]; other=pref[k]*suffix
                axes=tuple(1+h for h,v in enumerate(scope) if v not in sep[j])
                ctx[j],ls=norm(other.sum(axis=axes)); cs[j]=ps[k]+ss+ls
                suffix=suffix*_align(msg[j],sep[j],scope); ss=ss+scales[j]
    p = np.empty(n); p[order] = out
    if not np.isfinite(p).all():
        raise FloatingPointError('nonfinite quadrature result')
    return p
