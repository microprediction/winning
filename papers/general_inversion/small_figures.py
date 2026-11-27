"""Short checks behind in-text figures of the paper that had no command.

Each section prints the figures of one passage. Seeds are fixed in the
code and named in the section headers. Everything runs in seconds except
section G6, which draws a four-million-path referee (about a minute).

    python small_figures.py            # all sections
    python small_figures.py G1 G3      # chosen sections (G1 G2 G3 G6)
"""
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")

from winning.factor.core import hermite_nodes, jacobian_vector_product
from winning.factor.polish import _raw_and_derivative, race_jacobian
from winning.factor.races import (_setup, forward_grid, race_probabilities,
                                  student_base)

want = set(sys.argv[1:])


def on(tag):
    return not want or tag in want


def fd_jacobian(mu, h=1e-5, **kw):
    n = len(mu)
    out = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        out[:, j] = (race_probabilities(mu + e, **kw)
                     - race_probabilities(mu - e, **kw)) / (2 * h)
    return out


def continuum_jacobian(mu, V, D, base, points):
    """J0: off-diagonals from the lattice, diagonal imposed by the zero
    row sum, no normalization correction."""
    mu_s, V_s, D_s, F, W, fn, left, right = _setup(mu, V, D, None, None, base)
    sd = np.sqrt(D_s)
    x, _ = forward_grid(mu_s[None, :] + F @ V_s.T, sd, V_s, fn, left, right,
                        points)
    a, A, _ = _raw_and_derivative(mu_s, V_s, D_s, F, W, fn, sd, x, x[1] - x[0])
    J0 = A / a.sum()
    np.fill_diagonal(J0, 0.0)
    np.fill_diagonal(J0, -J0.sum(axis=1))
    return J0


if on("G1"):
    print("== G1 (Section 1.1 footnote): equicorrelation prices as independent "
          "noise; default_rng(0), n=8 ==")
    rng = np.random.default_rng(0)
    n = 8
    mu = rng.standard_normal(n) * 0.7
    mu -= mu.mean()
    worst = 0.0
    for s2, rho in ((1.7, 0.62), (0.4, 0.05), (2.5, 0.95)):
        V = np.full((n, 1), np.sqrt(s2 * rho))
        d = np.abs(race_probabilities(mu, V=V, D=np.full(n, s2 * (1 - rho)))
                   - race_probabilities(mu, D=np.full(n, s2 * (1 - rho)))).max()
        worst = max(worst, d)
        print(f"  s2={s2}, rho={rho}: max |p_equicorrelated - p_independent| {d:.1e}")
    print(f"  worst {worst:.1e}")

if on("G2"):
    print("== G2 (Section 3.1): window asked for by delta = 1e-12 in a "
          "six-runner Student-t(4) race; closed form, no random numbers ==")
    from scipy.stats import t as student_t
    nu, m, delta = 4, 6, 1e-12
    scale = np.sqrt((nu - 2) / nu)          # unit-variance standardization
    lower = scale * student_t.ppf(delta / m, nu)
    upper = scale * student_t.isf(delta ** (1 / m), nu)
    print(f"  lower delta-quantile of the winning time {lower:.0f} standard "
          f"units; window {upper - lower:.0f} wide; spacing at 257 points "
          f"{(upper - lower) / 256:.1f}")

if on("G2"):
    print("== G2 (Section 3.1): lattice error under a Student-t(2.5) base, "
          "n=40, against a 100,001-point lattice; default_rng(0) ==")
    rng = np.random.default_rng(0)
    n = 40
    mu = rng.standard_normal(n)
    mu -= mu.mean()
    st = student_base(2.5)
    ref = race_probabilities(mu, D=np.ones(n), base=st, points=100001)
    for pts in (257, 513, 1025):
        p = race_probabilities(mu, D=np.ones(n), base=st, points=pts)
        print(f"  {pts:5d} points: TV {0.5 * np.abs(p - ref).sum():.1e}")

if on("G3"):
    print("== G3a (Section 3.2 footnote): Jacobian against central differences; "
          "default_rng(3), n=8, rank one ==")
    rng = np.random.default_rng(3)
    n = 8
    mu = rng.standard_normal(n) * 0.6
    V = rng.standard_normal((n, 1)) * 0.4
    D = np.full(n, 0.6)
    for label, base, pts in (("Gaussian, 65 points", "normal", 65),
                             ("Gaussian, 257 points", "normal", 257),
                             ("Student-t(4), 65 points", student_base(4), 65),
                             ("Student-t(4), 257 points", student_base(4), 257)):
        fd = fd_jacobian(mu, V=V, D=D, points=pts, base=base)
        J = race_jacobian(mu, V=V, D=D, points=pts, base=base)
        J0 = continuum_jacobian(mu, V, D, base, pts)
        s = np.abs(fd).max()
        print(f"  {label:26s} J {np.abs(J - fd).max() / s:.1e}   "
              f"J0 {np.abs(J0 - fd).max() / s:.1e}   (relative to the largest entry)")

    print("== G3b (Section 3.2): column and row sums of J; default_rng(5), n=12 ==")
    rng = np.random.default_rng(5)
    n = 12
    mu = rng.normal(0, 0.8, n)
    mu -= mu.mean()
    g = rng.normal(0, 0.6, n)
    g -= g.mean()
    V, D = g[:, None], 0.5 + 0.5 * rng.random(n)
    for base in ("normal", "gumbel"):
        for pts in (25, 65, 257):
            J = race_jacobian(mu, V=V, D=D, base=base, points=pts)
            print(f"  {base:7s} {pts:4d} points: max |column sum| "
                  f"{np.abs(J.sum(axis=0)).max():.1e}   max |row sum| "
                  f"{np.abs(J.sum(axis=1)).max():.1e}   max |J_ij - J_ji| "
                  f"{np.abs(J - J.T).max():.1e}")

    print("== G3c (Section 5.2): directional derivative against central "
          "differences, with and without the normalization quotient ==")
    F, W = hermite_nodes(1, 15)
    h = np.random.default_rng(7).standard_normal(n)
    for pts in (25, 65, 101, 257):
        e = 1e-5 * h
        fd = (race_probabilities(mu + e, V=V, D=D, F=F, W=W, points=pts)
              - race_probabilities(mu - e, V=V, D=D, F=F, W=W, points=pts)) / 2e-5
        with_q = jacobian_vector_product(mu, V, D, F, W, h, points=pts,
                                         form="grid", normalized=True)
        without = jacobian_vector_product(mu, V, D, F, W, h, points=pts,
                                          form="grid", normalized=False)
        print(f"  L={pts:4d}: with quotient {np.abs(with_q - fd).max():.1e}   "
              f"without {np.abs(without - fd).max():.1e}")

if on("G6"):
    print("== G6 (Sections 4.1, 4.2): fixed-rule block kernel at cluster "
          "sharpness 18 against a four-million-path referee; default_rng(0) ==")
    from winning.factor.blocks import block_race_probabilities
    rng = np.random.default_rng(0)
    n, paths = 24, 4_000_000
    lab = np.repeat(np.arange(4), 6)
    mu = rng.standard_normal(n) * 0.6
    mu -= mu.mean()
    loading, D = np.full(n, 4.0), np.full(n, 0.05)
    p = block_race_probabilities(mu, lab, loading, D)
    wins = np.zeros(n)
    sim = np.random.default_rng(1)
    for _ in range(40):
        m = paths // 40
        x = (mu + loading * sim.standard_normal((m, 4))[:, lab]
             + np.sqrt(D) * sim.standard_normal((m, n)))
        wins += np.bincount(np.argmin(x, axis=1), minlength=n)
    ref = wins / paths
    print(f"  sharpness {4.0 / np.sqrt(0.05):.1f}; TV against the referee "
          f"{0.5 * np.abs(p - ref).sum():.1e}; group shares kernel "
          f"{np.round([p[lab == c].sum() for c in range(4)], 3)} referee "
          f"{np.round([ref[lab == c].sum() for c in range(4)], 3)}")
