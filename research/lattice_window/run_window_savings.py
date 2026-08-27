"""How many lattice points does the winner-bulk window save?

The package's window spans min-to-max of all abilities, so hopeless runners
stretch the grid and dilute resolution where races are decided. The winner-
bulk window (qpo's adaptive construction) sizes the lattice by the
distribution of the MINIMUM itself: G(x) = 1 - prod_j S_j(x) is the winner
cdf, and [G^-1(delta), G^-1(1-delta)] contains all but 2*delta of every
integrand -- including each hopeless runner's own win integrand, since even a
hopeless runner only wins by running a winner-bulk time.
"""
import numpy as np
from scipy.special import ndtr

def field_race(mu, sd, lo, hi, points):
    x = np.linspace(lo, hi, points); dx = x[1]-x[0]
    z = (x[None,:]-mu[:,None])/sd[:,None]
    S = np.maximum(1.0-ndtr(z), 1e-300)          # min-wins survival
    logS = np.log(S)
    f = np.exp(-0.5*z*z)/(sd[:,None]*np.sqrt(2*np.pi))
    rest = np.exp(np.clip(logS.sum(0)[None,:]-logS, -700, 0))
    p = (f*rest).sum(1)*dx
    return p/p.sum()

def winner_bulk(mu, sd, delta=1e-10):
    # bisection on G(x) = 1 - prod S_j
    def G(x):
        return 1.0-np.exp(np.sum(np.log(np.maximum(1.0-ndtr((x-mu)/sd),1e-300))))
    lo, hi = mu.min()-9*sd.max(), mu.max()+9*sd.max()
    a, b = lo, hi
    for _ in range(80):
        m_=(a+b)/2
        if G(m_)<delta: a=m_
        else: b=m_
    xlo=a
    a, b = lo, hi
    for _ in range(80):
        m_=(a+b)/2
        if G(m_)<1-delta: a=m_
        else: b=m_
    return xlo, b

rng = np.random.default_rng(0)
for label, n, n_hopeless, spread in [("30 live + 70 hopeless", 100, 70, 6.0),
                                     ("30 live + 470 hopeless", 500, 470, 8.0),
                                     ("all live (control)", 100, 0, 0.0)]:
    n_live = n-n_hopeless
    mu = np.r_[rng.normal(0,0.8,n_live), 2.0+spread*rng.random(n_hopeless)]
    sd = 0.5+0.5*rng.random(n)
    full_lo, full_hi = (mu-8*sd.max()).min(), (mu+8*sd.max()).max()
    blo, bhi = winner_bulk(mu, sd)
    ref = field_race(mu, sd, blo, bhi, 8193)
    print("%s: full window %.1f wide, winner bulk %.1f (%.1fx narrower)" % (
        label, full_hi-full_lo, bhi-blo, (full_hi-full_lo)/(bhi-blo)))
    print("   %8s %14s %14s" % ("points", "full-window TV", "bulk-window TV"))
    for pts in (33, 65, 129, 257, 513):
        e_full = 0.5*np.abs(field_race(mu, sd, full_lo, full_hi, pts)-ref).sum()
        e_bulk = 0.5*np.abs(field_race(mu, sd, blo, bhi, pts)-ref).sum()
        print("   %8d %14.2e %14.2e" % (pts, e_full, e_bulk))
