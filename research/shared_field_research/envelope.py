"""Research: exact shared lower-envelope preintegration for Gaussian races."""
import math
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
try:
    from numba import njit
except ImportError:
    def njit(f):
        return f

@njit
def _mass(l, r):
    if l >= 0:
        return .5*(math.erfc(l/math.sqrt(2))-math.erfc(r/math.sqrt(2)))
    return .5*(math.erfc(-r/math.sqrt(2))-math.erfc(-l/math.sqrt(2)))

@njit
def scan(intercepts, slopes):
    """Slopes sorted decreasingly; all conditional credits, summed over rows."""
    n = len(slopes)
    out = np.zeros(n)
    stack = np.empty(n, np.int64)
    left = np.empty(n)
    for row in intercepts:
        k = 0
        i = 0
        while i < n:
            j = i+1
            best = i
            while j < n and slopes[j] == slopes[i]:
                if row[j] < row[best]:
                    best = j
                j += 1
            x = -np.inf
            while k:
                h = stack[k-1]
                x = (row[best]-row[h])/(slopes[h]-slopes[best])
                if x > left[k-1]:
                    break
                k -= 1
            if k == 0:
                x = -np.inf
            stack[k] = best
            left[k] = x
            k += 1
            i = j
        for j in range(k):
            h = stack[j]
            r = left[j+1] if j+1 < k else np.inf
            credit = _mass(left[j], r)
            lo = h
            while lo > 0 and slopes[lo-1] == slopes[h]:
                lo -= 1
            hi = h+1
            while hi < n and slopes[hi] == slopes[h]:
                hi += 1
            count = 0
            for i in range(lo, hi):
                if row[i] == row[h]:
                    count += 1
            for i in range(lo, hi):
                if row[i] == row[h]:
                    out[i] += credit/count
    return out

def prepare(mu, A):
    """Explicit covariance square root; no rank cutoff or ridge."""
    mu = np.asarray(mu, float)
    A = np.asarray(A, float)
    if mu.ndim != 1 or not len(mu) or A.ndim != 2 or len(A) != len(mu) or not np.isfinite(mu).all() or not np.isfinite(A).all():
        raise ValueError('finite mu and explicit factor A required')
    mu = mu-mu[0]
    A = A-A[0]
    if not np.isfinite(mu).all() or not np.isfinite(A).all():
        raise ValueError('contrast overflow')
    if not A.shape[1] or not np.any(A):
        return mu, np.zeros(len(mu)), np.zeros((len(mu),0)), np.arange(len(mu))
    _, _, vt = np.linalg.svd(A-A.mean(axis=0), full_matrices=True)
    B = A@vt.T
    order = np.argsort(-B[:,0], kind='stable')
    return mu[order], B[order,0].copy(), B[order,1:].copy(), order

def shares(prepared, m=12, seed=0, batch=4096, direct=False):
    """2**m scrambled Sobol draws; direct=True is matched-rotation baseline."""
    mu,slope,resid,order = prepared
    n,r = resid.shape
    out = np.zeros(n)
    if r == 0 and not direct:
        out = scan(mu[None,:],slope)
    else:
        d = r+int(direct)
        gen = qmc.Sobol(d, scramble=True, seed=seed)
        N = 2**m
        for start in range(0,N,batch):
            u = gen.random(min(batch,N-start))
            z = ndtri(np.clip(u, np.finfo(float).tiny, 1-np.finfo(float).eps))
            if direct:
                x = mu+z[:,1:]@resid.T+z[:,:1]*slope
                ties = x == x.min(axis=1,keepdims=True)
                out += (ties/ties.sum(axis=1,keepdims=True)).sum(axis=0)
            else:
                out += scan(mu+z@resid.T,slope)
        out /= N
    p = np.empty(n); p[order] = out
    return p
