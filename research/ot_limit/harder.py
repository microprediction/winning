"""Closing three gaps in extremal_harmonic_analysis.tex.

(A) Theorem 3.1 was only really verified at r=2. Here it is specialised to r=3
    in closed form and tested with a NONTRIVIAL zonal field, exposure threshold
    active, against an exact 1-D reduction of the continuum argmax. Comparison
    is by Legendre moments E[P_l(z*)], which are integrals and therefore immune
    to binning error.

(B) Open Problem 2, the noisy transfer function lambda_k(sigma). Measured by
    driving the eigenvector directly (mu = a cos k theta) with a paired
    +-a design and the deterministic transform, which is far better conditioned
    than finite-differencing a single Jacobian column.

(C) Open Problem 1, nonuniform site density. The paper conjectured a weighted
    Laplacian with a drift term. The finite-volume computation says otherwise:
    the facet conductance is k_i = phi(0)/(2 Delta_i), so k_i Delta_i is
    CONSTANT and the density q cancels. Prediction: rho = 1 - sqrt(pi/2) mu''
    no matter how the competitors are sampled.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre
from scipy.stats import norm

import circle_spectral as C
import soft_laguerre as SL

SQRT_2PI = np.sqrt(2 * np.pi)
C2 = np.sqrt(np.pi / 2)
_DP = {2: (lambda t: 3 * t, lambda t: 3 * np.ones_like(t)),
       4: (lambda t: (35 * t ** 3 - 15 * t) / 2,
           lambda t: (105 * t ** 2 - 15) / 2)}


# =============================================================== (A) r = 3

def r3_density(z, a, ell, ny=401, nphi=201):
    """Theorem 3.1 at r=3 for mu = a P_l(v.e).

    d=2 so det(aI-H) = a^2 - a e_1 + e_2, and the three radial integrals are
    elementary:  int_A^inf e^{-a^2/2}(a^2 - a e_1 + e_2) da
                 = (A - e_1) e^{-A^2/2} + (1 + e_2) sqrt(2pi) Phibar(A).
    Zonal frame: H = diag((1-z^2)f'' - z f', -z f'), |grad|^2 = (1-z^2) f'^2.
    """
    dP, d2P = _DP[ell]
    f = lambda t: a * eval_legendre(ell, t)
    fp, fpp = a * dP(z), a * d2P(z)
    Htt = (1 - z ** 2) * fpp - z * fp
    Hpp = -z * fp
    e1, e2 = Htt + Hpp, Htt * Hpp
    grad2 = (1 - z ** 2) * fp ** 2

    # A(v) = sup over w. Parameterise w by (y, t) = (w.e, w.v); the Gram
    # constraint gives t in z y +- sqrt((1-z^2)(1-y^2)).
    y = np.linspace(-1, 1, ny)
    cph = np.cos(np.linspace(0, np.pi, nphi))
    A = np.empty_like(z)
    for i, zi in enumerate(z):
        sv = np.sqrt(max(1 - zi ** 2, 0.0))
        sw = np.sqrt(np.maximum(1 - y ** 2, 0.0))
        t = zi * y[:, None] + sv * sw[:, None] * cph[None, :]
        num = (f(y)[:, None] - f(np.array([zi]))[0]
               - fp[i] * (y[:, None] - zi * t))
        den = 1 - t
        A[i] = max(np.max(np.where(den > 1e-8, num / np.where(den > 1e-8, den, 1),
                                   -np.inf)), Htt[i], Hpp[i])
    rho = np.exp(-grad2 / 2) * ((A - e1) * np.exp(-A ** 2 / 2)
                                + (1 + e2) * SQRT_2PI * norm.sf(A))
    return rho, A


def r3_montecarlo(a, ell, draws=6_000_000, npsi=8192, seed=5):
    """Exact continuum argmax on S^2 for a zonal field, by 1-D reduction:
    U depends on v only through (v.e, v.F) and for fixed v.e the maximum of
    v.F is attained in span{e,F}, so the maximiser lies on that great circle."""
    rng = np.random.default_rng(seed)
    psi = 2 * np.pi * np.arange(npsi) / npsi
    mu_psi = a * eval_legendre(ell, np.cos(psi))
    out = np.empty(draws)
    done, chunk = 0, 20000
    while done < draws:
        m = min(chunk, draws - done)
        F = rng.standard_normal((m, 3))
        R = np.linalg.norm(F, axis=1)
        alpha = np.arccos(np.clip(F[:, 0] / R, -1, 1))
        U = mu_psi[None, :] + R[:, None] * np.cos(psi[None, :] - alpha[:, None])
        out[done:done + m] = np.cos(psi[U.argmax(1)])
        done += m
    return out


def part_A():
    print("\n(A) Theorem 3.1 at r=3, nontrivial zonal field, exposure ACTIVE")
    print("    comparison by Legendre moments E[P_l(z*)] (binning-free)")
    z = np.linspace(-0.99995, 0.99995, 1201)
    for a, ell in ((0.15, 2), (0.40, 2), (0.15, 4)):
        rho, A = r3_density(z, a, ell)
        rho = rho / (np.trapezoid(rho, z) / 2.0)
        s = r3_montecarlo(a, ell)
        print(f"   a={a:4.2f} l={ell}:  A in [{A.min():+.3f},{A.max():+.3f}]"
              f"   mass check {np.trapezoid(rho, z)/2:.6f}")
        worst = 0.0
        for L in (1, 2, 3, 4, 6):
            pred = np.trapezoid(eval_legendre(L, z) * rho, z) / 2.0
            meas = eval_legendre(L, s).mean()
            se = eval_legendre(L, s).std() / np.sqrt(len(s))
            worst = max(worst, abs(pred - meas))
            print(f"      P_{L}: predicted {pred:+.5f}   measured {meas:+.5f}"
                  f"   diff {pred-meas:+.5f}  (mc se {se:.5f})")
        print(f"      -> worst moment discrepancy {worst:.5f}")


# ====================================================== (B) transfer function

def part_B(N=512, Q=61, points=20001, a=2e-3):
    print("\n(B) noisy transfer function lambda_k(sigma), eigenvector driving")
    th, V = C.geometry(N)
    F, W = SL.gh_nodes(2, Q)
    ks = (1, 2, 4, 8, 16, 32)
    print(f"   {'sigma':>7} " + " ".join(f"{'k='+str(k):>9}" for k in ks))
    table = {}
    for sig in (0.0316, 0.0707, 0.1414, 0.2828, 0.5657):
        D = np.full(N, sig ** 2)
        row = []
        for k in ks:
            mu = a * np.cos(k * th)
            pp = SL.shares_gh(mu, V, D, F, W, points)
            pm = SL.shares_gh(-mu, V, D, F, W, points)
            g = C.cos_amp(N * (pp - pm) / 2, th, k) / a
            row.append(g / (C2 * k * k))
        table[sig] = np.array(row)
        print(f"   {sig:7.4f} " + " ".join(f"{v:9.4f}" for v in row))

    print("\n   collapse quality (residual scatter about a smooth fit; lower is better)")
    best = None
    for fn, name in ((lambda s, k: s * k * k, "sigma k^2"),
                     (lambda s, k: s * k, "sigma k"),
                     (lambda s, k: s * s * k * k, "sigma^2 k^2"),
                     (lambda s, k: s * k ** 1.5, "sigma k^1.5")):
        xs, ys = [], []
        for s in table:
            for i, k in enumerate(ks):
                v = table[s][i]
                if 0.02 < v < 0.98:
                    xs.append(fn(s, k)); ys.append(v)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 8:
            continue
        c = np.polyfit(np.log(xs), np.log(ys), 3)
        sc = (np.log(ys) - np.polyval(c, np.log(xs))).std()
        print(f"      {name:>12}: n={len(xs):3d}  scatter {sc:.4f}")
        if best is None or sc < best[1]:
            best = (name, sc)
    print(f"   -> best collapse variable: {best[0]}")
    return table


# ================================================= (C) nonuniform site density

def nonuniform_sites(N, amp):
    """Sites with angular density q ~ 1 + amp cos(theta), by inverse cdf."""
    u = (np.arange(N) + 0.5) / N
    g = np.linspace(0, 2 * np.pi, 400001)
    return np.interp(u, (g + amp * np.sin(g)) / (2 * np.pi), g)


def part_C(N=1536, a=0.01, draws=8_000_000):
    print("\n(C) does the sampling density change the continuum operator?")
    print("    conductance k_i = phi(0)/(2 Delta_i) gives k_i Delta_i = const,")
    print("    so q should cancel and the operator stay -sqrt(pi/2) d^2/dtheta^2")
    print(f"   {'k':>3} {'dens amp':>9} {'min/max gap':>12} {'gain':>9} "
          f"{'c_2 k^2':>9} {'ratio':>8}")
    for k in (2, 3):
        for amp in (0.0, 0.4, 0.8):
            th = nonuniform_sites(N, amp)
            V = np.column_stack([np.cos(th), np.sin(th)])
            mu = a * np.cos(k * th)
            rng = np.random.default_rng(2)
            cnt, done = np.zeros(N), 0
            while done < draws:
                m = min(4000, draws - done)
                cnt += np.bincount(
                    (mu + rng.standard_normal((m, 2)) @ V.T).argmax(1),
                    minlength=N)
                done += m
            gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
            arc = (gaps + np.roll(gaps, 1)) / 2
            wq = arc / (2 * np.pi)
            rho = (cnt / done) / wq
            gain = 2 * np.sum((rho - 1) * np.cos(k * th) * wq) / a
            print(f"   {k:3d} {amp:9.2f} {gaps.min()/gaps.max():12.3f} "
                  f"{gain:9.3f} {C2*k*k:9.3f} {gain/(C2*k*k):8.4f}")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
