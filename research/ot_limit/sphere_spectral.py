"""Is the photo-finish operator the Laplace-Beltrami operator on S^{r-1}?

PREDICTION under test. Put possible winners densely on the unit sphere in
factor space, v_i in S^{r-1}, and race

    U(v) = mu(v) + F'v,     F ~ N(0, I_r).

Write F = R n with n uniform on the sphere. At mu = 0 the winner is n itself.
For small mu the winner displaces tangentially by grad_S mu(n)/R, and pushing
the uniform spherical measure through that map gives

    rho(v) = 1 - c_r * Laplace_S mu(v) + o(mu),
    c_r = E[1/R] = 2^{-1/2} Gamma((r-1)/2) / Gamma(r/2).

Zonal harmonics of degree l satisfy -Laplace_S Y_l = l(l + r - 2) Y_l, so the
FORWARD GAIN of degree l should be

    g(l, r) = c_r * l (l + r - 2).

r = 2 gives c_2 = sqrt(pi/2) and l^2, which is the circle result already
confirmed. This script tests r = 3, 4, 5, where the eigenvalue is l(l+1),
l(l+2), l(l+3) and the constants are 0.7979, 0.6267, 0.5319.

METHOD. Sites are exactly uniform (normalized Gaussians). The test field is
the zonal Gegenbauer polynomial C_l^{(r-2)/2}(v . e), an exact eigenfunction.
Gains are measured with a PAIRED +a / -a design sharing common random
numbers: the leading artifact at finite N is the local site-density
fluctuation (a site in a sparse patch wins more often), which is identical in
both arms and cancels exactly in the difference, as do all even orders in a.
So the estimator isolates the linear response.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_chebyt, eval_gegenbauer, gammaln


def c_r(r):
    """E[1/R] for R = |F|, F ~ N(0, I_r)."""
    return np.exp(-0.5 * np.log(2.0) + gammaln((r - 1) / 2) - gammaln(r / 2))


def zonal(r, l, z):
    """Zonal harmonic of degree l on S^{r-1}, evaluated at cos(polar angle)."""
    if r == 2:
        return eval_chebyt(l, np.clip(z, -1, 1))       # cos(l theta)
    return eval_gegenbauer(l, (r - 2) / 2.0, np.clip(z, -1, 1))


def sphere_sites(N, r, rng):
    V = rng.standard_normal((N, r))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def paired_gain(V, Y, a, draws, rng, chunk=1500):
    """Linear-response gain from a +a/-a paired race with common F draws."""
    N, r = V.shape
    mu_p, mu_m = a * Y, -a * Y
    cp = np.zeros(N)
    cm = np.zeros(N)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        base = rng.standard_normal((m, r)) @ V.T
        cp += np.bincount((base + mu_p).argmax(1), minlength=N)
        cm += np.bincount((base + mu_m).argmax(1), minlength=N)
        done += m
    mp = N * (cp / draws) - 1.0
    mm = N * (cm / draws) - 1.0
    return float((mp - mm) @ Y / (2 * a * (Y @ Y)))


def run(rs=(2, 3, 4, 5), ls=(1, 2, 3, 4), N=30000, a=0.01, draws=60_000,
        seed=17):
    print(f"N={N} sites, amplitude a={a}, {draws:,} paired draws\n")
    print(f"{'r':>3} {'l':>3} {'eig l(l+r-2)':>13} {'c_r':>8} "
          f"{'predicted':>10} {'measured':>10} {'ratio':>8}")
    out = {}
    for r in rs:
        rng = np.random.default_rng(seed + r)
        V = sphere_sites(N, r, rng)
        z = V[:, 0]
        cr = c_r(r)
        for l in ls:
            if r == 5 and l == 4:
                continue                        # coverage too coarse in S^4
            Y = zonal(r, l, z)
            Y = Y - Y.mean()
            eig = l * (l + r - 2)
            pred = cr * eig
            meas = paired_gain(V, Y, a, draws, rng)
            out[(r, l)] = (pred, meas)
            print(f"{r:3d} {l:3d} {eig:13d} {cr:8.4f} {pred:10.4f} "
                  f"{meas:10.4f} {meas/pred:8.3f}")
        print()
    return out


def convergence_in_N(r=3, l=2, Ns=(4000, 10000, 30000, 80000), a=0.01,
                     draws=60_000, seed=5):
    """Does the measured gain approach the continuum prediction as N grows?"""
    pred = c_r(r) * l * (l + r - 2)
    print(f"discretization check, r={r}, l={l}, prediction {pred:.4f}")
    for N in Ns:
        rng = np.random.default_rng(seed + N)
        V = sphere_sites(N, r, rng)
        Y = zonal(r, l, V[:, 0])
        Y = Y - Y.mean()
        g = paired_gain(V, Y, a, draws, rng)
        print(f"   N={N:6d}   measured {g:8.4f}   ratio {g/pred:.3f}")


if __name__ == "__main__":
    print("c_r check (E[1/R], chi with r dof):",
          {r: round(float(c_r(r)), 5) for r in (2, 3, 4, 5)})
    print("  c_2 vs sqrt(pi/2):", float(c_r(2)), np.sqrt(np.pi / 2), "\n")
    run()
    convergence_in_N()
