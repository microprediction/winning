"""The exact nonlinear winner density on the circle, and its verification.

THE FORMULA. For the race U(theta) = mu(theta) + F'v(theta) on S^1 with
F ~ N(0, I_2), the density of the global argmax is exactly

    rho(theta) = 2 pi p(theta)
               = e^{-mu'(theta)^2/2} [ e^{-A(theta)^2/2}
                                       - sqrt(2 pi) mu''(theta) Phibar(A(theta)) ]

where Phibar = 1 - Phi and A is the EXPOSURE THRESHOLD

    A(theta) = sup_{d != 0} [mu(theta+d) - mu(theta) - mu'(theta) sin d] / (1 - cos d),

whose d -> 0 limit is mu''(theta) (so that value is a candidate in the sup).

WHERE IT COMES FROM. Writing F = R n, a direction theta can only win if the
tangential stationarity mu'(theta) = R sin(theta - Theta) holds, which pins the
tangential part of F and leaves only the radial component a = F.v free. The
Jacobian of (theta, a) -> F is a - mu''(theta), the Gaussian weight is
exp(-(a^2 + mu'^2)/2), and theta beats every competitor exactly when
a >= A(theta). Integrating a over [A, infinity) gives the formula in closed
form because both integrals are elementary. In r dimensions the same argument
gives det(aI - Hess mu) in place of (a - mu''), which is the Gaussian
Minkowski / Monge-Ampere surface density.

WHY IT MATTERS HERE. Setting A == 0 recovers the candidate operator
e^{-mu'^2/2}(1 - sqrt(pi/2) mu'') that this repo FALSIFIED earlier (it is
first-order correct, second-order wrong by a^2 cos^2/2 at k=1). The exposure
threshold is precisely the missing piece, and it is also the object that
carries all the global/locking behaviour.

Run: python nonlinear_circle.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

import circle_spectral as C

SQRT_2PI = np.sqrt(2 * np.pi)


def exposure(mu, mup, mupp):
    """A(theta), by direct evaluation of the sup over shifts on the grid."""
    N = len(mu)
    d = 2 * np.pi * np.arange(N) / N
    den = 1 - np.cos(d)
    idx = (np.arange(N)[:, None] + np.arange(N)[None, :]) % N
    num = mu[idx] - mu[:, None] - mup[:, None] * np.sin(d)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = np.where(den > 1e-12, num / np.maximum(den, 1e-300), -np.inf)
    return np.maximum(mupp, rat.max(axis=1))


def rho_exact(mu, mup, mupp):
    A = exposure(mu, mup, mupp)
    return np.exp(-mup**2 / 2) * (np.exp(-A**2 / 2)
                                  - SQRT_2PI * mupp * norm.sf(A))


def rho_linear(mupp):
    """The first-order law: 1 - sqrt(pi/2) mu''."""
    return 1.0 - np.sqrt(np.pi / 2) * mupp


def harmonic(a, k, N=512):
    th = 2 * np.pi * np.arange(N) / N
    return th, a * np.cos(k * th), -a * k * np.sin(k * th), \
        -a * k * k * np.cos(k * th)


def check_projected_normal():
    """k = 1 is exactly solvable: mu = a cos theta shifts F1 by a, so the
    winner law is the ANGULAR density of N((a,0), I_2), the projected normal.
    Claim: the sup defining A is constant in the shift, giving A = -a cos."""
    print("k=1 anchor: exposure threshold and the projected normal")
    for a in (0.2, 0.7, 1.5):
        th, mu, mup, mupp = harmonic(a, 1)
        A = exposure(mu, mup, mupp)
        c = a * np.cos(th)
        pn = np.exp(-a**2 / 2) * (1 + c * SQRT_2PI * np.exp(c**2 / 2)
                                  * norm.cdf(c))
        print(f"   a={a:4.1f}  max|A - (-a cos)| = {np.abs(A + c).max():.2e}"
              f"   max|formula - projected normal| = "
              f"{np.abs(rho_exact(mu, mup, mupp) - pn).max():.2e}")


def check_monte_carlo(N=512, draws=3_000_000):
    print(f"\nformula vs Monte Carlo ({draws:,} races, N={N} sites)")
    print(f"{'a':>6}{'k':>3}{'a k^2':>8} {'mean rho':>10} {'gain exact':>11} "
          f"{'gain MC':>9} {'gain linear':>12} {'G = ex/lin':>11}")
    _, V = C.geometry(N)
    for a, k in ((0.05, 1), (0.3, 1), (1.0, 1), (0.02, 3), (0.05, 3),
                 (0.15, 3), (0.02, 8), (0.05, 8)):
        th, mu, mup, mupp = harmonic(a, k, N)
        rf = rho_exact(mu, mup, mupp)
        rm = N * C.shares_mc(mu, V, 0.0, draws, seed=4)
        gf = C.cos_amp(rf - 1, th, k) / a
        gm = C.cos_amp(rm - 1, th, k) / a
        gl = np.sqrt(np.pi / 2) * k * k
        print(f"{a:6.2f}{k:3d}{a*k*k:8.3f} {rf.mean():10.6f} {gf:11.3f} "
              f"{gm:9.3f} {gl:12.3f} {gf/gl:11.3f}")


def universal_G(N=512):
    """The empirically-measured universal G(a k^2) is now a consequence:
    it is the exact gain divided by the linear gain."""
    print("\nG(a k^2) from the exact formula -- collapse across (a, k)")
    print(f"{'a k^2':>8} {'a':>7}{'k':>4} {'G':>9}")
    rows = []
    for a in (0.005, 0.02, 0.08):
        for k in (2, 4, 8, 16):
            th, mu, mup, mupp = harmonic(a, k, N)
            g = C.cos_amp(rho_exact(mu, mup, mupp) - 1, th, k) / a
            rows.append((a * k * k, a, k, g / (np.sqrt(np.pi / 2) * k * k)))
    for x, a, k, G in sorted(rows):
        print(f"{x:8.3f} {a:7.3f}{k:4d} {G:9.4f}")


if __name__ == "__main__":
    check_projected_normal()
    check_monte_carlo()
    universal_G()
