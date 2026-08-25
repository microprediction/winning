"""Does the factor-probit transform converge to Gaussian semi-discrete OT?

Claim under test (rank r=2, where everything is closed form):

  The max-wins race U_i = mu_i + v_i'F, F ~ N(0, I_r), assigns F to its
  winner by a POWER (Laguerre) DIAGRAM.  Exactly:

      C_i = {x : mu_i + v_i'x >= mu_j + v_j'x for all j}
          = {x : |x - z_i|^2 - w_i <= |x - z_j|^2 - w_j for all j},
      with   z_i = v_i / 2,     w_i = mu_i + |v_i|^2 / 4.

  (Expand the norms; |x|^2 cancels.)  A bijection, not an approximation.

  With Sigma_tau = V V' + tau^2 I:
      (A)  p_i^tau -> gamma_r(C_i)                     as tau -> 0
      (B)  dp_i/dmu_j -> -k_ij, the weighted graph Laplacian of the power
           diagram, k_ij = (1/|v_i - v_j|) int_{C_i cap C_j} phi_r dH^{r-1}
           -- which is the Newton Hessian of semi-discrete optimal transport.

  Why that edge weight: the i|j boundary is {x : n'x = c}, n = v_i - v_j,
  c = mu_j - mu_i.  Raising mu_i by delta slides that hyperplane a normal
  distance delta/|n| into C_j, so the flux is (1/|n|) times the Gaussian
  surface integral over the shared facet.

  In r=2 the facet integral is closed form: with x(t) = x_0 + t d,
  x_0 = c n/|n|^2, d unit and orthogonal to n, we get x_0 _|_ d, so
  |x(t)|^2 = |x_0|^2 + t^2 and

      int phi_2 dH^1 = phi_1(|c|/|n|) * (Phi(t_hi) - Phi(t_lo)),

  with [t_lo, t_hi] the facet after clipping against the other half-planes.

Three parts:
  1. RING: N sites on a circle with equal mu -- masses are exactly 1/N and
     the Laplacian is exactly a cycle graph.  Fully analytic reference.
  2. GENERAL: random sites; validate the closed-form facet weights against
     common-random-number finite differences of a Monte Carlo mass estimate.
  3. Which claim vs which method: compare the MATH limit (probit shares by
     Monte Carlo) against the ALGORITHM (probit shares by Gauss-Hermite
     quadrature + lattice), to see whether any gap is in (A)-(B) or in the
     numerics.

Run: python laguerre_limit.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

import fastrace
from winning.factor.core import hermite_nodes

SEED = 20260819


# ---------------------------------------------------------------- geometry

def facet_weight(i, j, mu, V):
    """k_ij = (1/|n|) int_{facet} phi_2, closed form; 0 if no shared facet."""
    n = V[i] - V[j]
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return 0.0
    c = mu[j] - mu[i]
    x0 = c * n / nn**2
    d = np.array([-n[1], n[0]]) / nn
    t_lo, t_hi = -np.inf, np.inf
    for k in range(len(mu)):
        if k in (i, j):
            continue
        A = float((V[i] - V[k]) @ d)
        B = float((V[i] - V[k]) @ x0 - (mu[k] - mu[i]))
        if abs(A) < 1e-14:
            if B < 0:
                return 0.0
        elif A > 0:
            t_lo = max(t_lo, -B / A)
        else:
            t_hi = min(t_hi, -B / A)
    if t_hi <= t_lo:
        return 0.0
    return float(norm.pdf(abs(c) / nn) * (norm.cdf(t_hi) - norm.cdf(t_lo)) / nn)


def laplacian_exact(mu, V):
    n = len(mu)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = facet_weight(i, j, mu, V)
    J = -K.copy()
    np.fill_diagonal(J, K.sum(axis=1))
    return J


# ------------------------------------------------------------------ probit

def probit_shares_gh(mu, V, tau, Q, points):
    """The package algorithm: Gauss-Hermite over F, lattice over the race."""
    F, W = hermite_nodes(V.shape[1], Q=Q)
    p, _ = fastrace.win_probabilities_factor(
        -np.asarray(mu, float), np.asarray(V, float),
        np.full(len(mu), tau**2), np.ascontiguousarray(F),
        np.ascontiguousarray(W), points)
    return p


def probit_shares_mc(mu, V, tau, draws=20_000_000, seed=11):
    """The same mathematical object, by brute force -- no quadrature."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(mu))
    done = 0
    while done < draws:
        m = min(2_000_000, draws - done)
        U = mu + rng.standard_normal((m, V.shape[1])) @ V.T
        if tau > 0:
            U = U + tau * rng.standard_normal((m, len(mu)))
        counts += np.bincount(U.argmax(axis=1), minlength=len(mu))
        done += m
    return counts / draws


def jacobian_fd(shares_fn, mu, h):
    n = len(mu)
    J = np.zeros((n, n))
    for j in range(n):
        mp, mm = mu.copy(), mu.copy()
        mp[j] += h
        mm[j] -= h
        J[:, j] = (shares_fn(mp) - shares_fn(mm)) / (2 * h)
    return J


# -------------------------------------------------------------------- ring

def part1_ring(N=7, a=1.0):
    th = 2 * np.pi * np.arange(N) / N
    V = a * np.column_stack([np.cos(th), np.sin(th)])
    mu = np.zeros(N)

    p_true = np.full(N, 1.0 / N)
    k_true = norm.pdf(0.0) * 0.5 / (2 * a * np.sin(np.pi / N))
    J_true = np.zeros((N, N))
    for i in range(N):
        for j in ((i + 1) % N, (i - 1) % N):
            J_true[i, j] = -k_true
    np.fill_diagonal(J_true, 2 * k_true)

    print(f"PART 1 -- ring of {N} sites, radius {a} (analytic reference)")
    print(f"  exact cell mass 1/N = {1/N:.6f}; exact edge weight "
          f"k = phi(0)/2 / (2a sin(pi/N)) = {k_true:.6f}")
    Jc = laplacian_exact(mu, V)
    print(f"  closed-form facet code reproduces it: max|J_code - J_true| = "
          f"{np.abs(Jc - J_true).max():.3e}")

    print(f"\n  {'tau':>7} {'Q':>4} {'points':>7} {'max|p-1/N|':>11} "
          f"{'max|J-Jtrue|':>13} {'rel':>7}")
    for tau in (0.5, 0.25, 0.125, 0.0625, 0.03125):
        for Q in (15, 41, 81):
            pts = int(np.clip(1000 / tau, 4000, 120_000))
            p = probit_shares_gh(mu, V, tau, Q, pts)
            J = jacobian_fd(lambda m: probit_shares_gh(m, V, tau, Q, pts),
                            mu, max(1e-3, 0.05 * tau))
            print(f"  {tau:7.5f} {Q:4d} {pts:7d} "
                  f"{np.abs(p - p_true).max():11.3e} "
                  f"{np.abs(J - J_true).max():13.3e} "
                  f"{np.abs(J - J_true).max()/(2*k_true):7.3f}")

    print("\n  same limit by Monte Carlo (no quadrature):")
    for tau in (0.5, 0.125, 0.03125, 0.0):
        p = probit_shares_mc(mu, V, tau)
        print(f"  {tau:7.5f}   MC  max|p-1/N| = {np.abs(p - p_true).max():.3e}"
              f"   (mc se ~ {np.sqrt(p_true[0]*(1-p_true[0])/2e7):.1e})")
    return V, mu


# ----------------------------------------------------------------- general

def part2_general(N=8, seed=SEED):
    rng = np.random.default_rng(seed)
    for attempt in range(50):
        V = rng.normal(scale=1.2, size=(N, 2))
        mu = rng.normal(scale=0.3, size=N)
        mu -= mu.mean()
        p = probit_shares_mc(mu, V, 0.0, draws=2_000_000, seed=3)
        if p.min() > 0.02:
            break
    print(f"\nPART 2 -- random configuration, all cells non-trivial "
          f"(min mass {p.min():.3f}, attempt {attempt+1})")
    J0 = laplacian_exact(mu, V)
    ev = np.linalg.eigvalsh(J0)
    adj = int((J0 < -1e-12).sum() / 2)
    print(f"  closed-form Laplacian: {adj} adjacent pairs of {N*(N-1)//2}, "
          f"rows sum {np.abs(J0.sum(1)).max():.1e}, eig min {ev[0]:+.1e}, "
          f"second {ev[1]:+.3e}")

    h = 0.02
    Jmc = jacobian_fd(lambda m: probit_shares_mc(m, V, 0.0,
                                                 draws=8_000_000, seed=5),
                      mu, h)
    err = np.abs(Jmc - J0).max()
    print(f"  validated against common-random-number finite differences of "
          f"the exact\n  Laguerre masses (h={h}, 8e6 draws): "
          f"max|J_mc - J_closedform| = {err:.3e} "
          f"({err/np.abs(J0).max()*100:.1f}% of scale)")
    return V, mu, J0


def part3_who_is_wrong(V, mu, J0):
    print("\nPART 3 -- is any gap in the mathematics or in the quadrature?")
    p0 = probit_shares_mc(mu, V, 0.0, draws=20_000_000, seed=9)
    print(f"  {'tau':>8} {'MC max|p-p0|':>14} {'GH max|p-p0|':>14} "
          f"{'GH-vs-MC':>10}")
    for tau in (0.5, 0.25, 0.125, 0.0625):
        pmc = probit_shares_mc(mu, V, tau, draws=20_000_000, seed=9)
        pgh = probit_shares_gh(mu, V, tau, 81,
                               int(np.clip(1000 / tau, 4000, 120_000)))
        print(f"  {tau:8.5f} {np.abs(pmc - p0).max():14.3e} "
              f"{np.abs(pgh - p0).max():14.3e} "
              f"{np.abs(pgh - pmc).max():10.3e}")


if __name__ == "__main__":
    part1_ring()
    V, mu, J0 = part2_general()
    part3_who_is_wrong(V, mu, J0)
