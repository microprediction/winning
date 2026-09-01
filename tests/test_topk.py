import numpy as np
import pytest


def test_k1_is_the_win_probability():
    """k = 1 must reproduce the race probabilities to machine
    precision; measured 1.7e-16 when pinned."""
    from winning.factor.topk import top_k_probabilities
    from winning.factor.races import race_probabilities

    mu = np.array([-0.5, -0.1, 0.2, 0.6])
    D = np.ones(4)
    q = top_k_probabilities(mu, 1, D=D)
    p = race_probabilities(mu, D=D)
    assert np.abs(q - p).max() < 1e-12


def test_complements_and_bottom_k():
    """P(in the best n-1) = 1 - P(worst), and bottom-k is the
    complement of top-(n-k), so no reflected base exists anywhere."""
    from winning.factor.topk import (top_k_probabilities,
                                     bottom_k_probabilities)
    from winning.factor.races import race_probabilities

    rng = np.random.default_rng(3)
    n = 30
    mu = rng.normal(0, 0.8, n)
    D = 0.5 + rng.random(n)
    worst = race_probabilities(-mu, D=D)
    q = top_k_probabilities(mu, n - 1, D=D)
    assert np.abs(q - (1 - worst)).max() < 1e-6
    b = bottom_k_probabilities(mu, 1, D=D)
    assert np.abs(b - worst).max() < 1e-6


def test_deconvolution_matches_direct_leave_one_out():
    """The stable-direction deconvolution must reproduce n separate
    leave-one-out dynamic programs; measured 1.4e-15 at n=120, k=30,
    location spread 2.0 when pinned."""
    from winning.factor.topk import (_count_window, _count_distribution,
                                     _leave_one_out_cdf)
    from winning.factor.races import BASES

    rng = np.random.default_rng(7)
    n, k = 80, 20
    mu = rng.normal(0, 2.0, n)
    sd = np.sqrt(0.3 + rng.random(n))
    base = BASES["normal"]
    lo, hi = _count_window(mu, sd, k, base)
    x = np.linspace(lo, hi, 257)
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, _, _ = base(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    C = _count_distribution(F)
    cdf = _leave_one_out_cdf(C, F, k)
    for i in rng.choice(n, 6, replace=False):
        Ci = _count_distribution(np.delete(F, i, axis=1))
        direct = Ci[:, :k].sum(axis=1)
        assert np.abs(cdf[i] - direct).max() < 1e-12


def test_monte_carlo_referee_correlated():
    """Rank-one factor field against an argpartition Monte Carlo; run
    with RuntimeWarnings promoted to errors so recursion overflow can
    never sneak back."""
    import warnings
    from winning.factor.topk import top_k_probabilities

    rng = np.random.default_rng(0)
    n, k = 10, 3
    mu = rng.normal(0, 0.7, n)
    mu -= mu.mean()
    D = 0.5 + rng.random(n)
    v = 0.5 * np.ones(n) + 0.2 * rng.random(n)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        q = top_k_probabilities(mu, k, V=v, D=D)
    M = 400_000
    f = rng.standard_normal(M)
    X = (mu[None, :] + np.outer(f, v - v.mean())
         + rng.standard_normal((M, n)) * np.sqrt(D)[None, :])
    thresh = np.partition(X, k - 1, axis=1)[:, k - 1]
    emp = (X <= thresh[:, None]).mean(axis=0)
    assert abs(q.sum() - k) < 1e-9
    assert 0.5 * np.abs(q - emp).sum() < 5e-3


def test_refusals_and_slot_check():
    from winning.factor.topk import top_k_probabilities, _checked_topk

    with pytest.raises(ValueError):
        top_k_probabilities(np.zeros(5), 5, D=np.ones(5))
    with pytest.raises(NotImplementedError, match="rank"):
        top_k_probabilities(np.zeros(6), 2, V=np.ones((6, 3)),
                            D=np.ones(6))
    with pytest.raises(RuntimeError, match="slots"):
        _checked_topk(np.full(6, 0.1), 3, "test race")


def test_rust_matches_numpy():
    """The compiled kernel and the numpy path must agree to machine
    precision; measured 6.7e-16 at n=200, k=60 when pinned."""
    import winning.factor.topk as T

    if not T._HAVE_RUST:
        pytest.skip("fastrace without top_k")
    rng = np.random.default_rng(0)
    n, k = 120, 40
    mu = rng.normal(0, 1.2, n)
    D = 0.4 + rng.random(n)
    q_rust = T.top_k_probabilities(mu, k, D=D)
    saved = T._HAVE_RUST
    T._HAVE_RUST = False
    try:
        q_np = T.top_k_probabilities(mu, k, D=D)
    finally:
        T._HAVE_RUST = saved
    assert np.abs(q_rust - q_np).max() < 1e-12


def test_jacobian_matches_finite_differences():
    """The cutoff tie-density Jacobian against central differences of
    the forward map (measured 6.7e-11 when pinned), with k=1 recovering
    the win-probability Jacobian to 1.2e-15, off-diagonals symmetric,
    columns summing to zero, and minus the matrix PSD -- the rank-k
    boundary Laplacian the potential's concavity demands."""
    from winning.factor.topk import top_k_jacobian, top_k_probabilities
    from winning.factor.polish import race_jacobian

    rng = np.random.default_rng(2)
    n, k = 6, 2
    mu = rng.normal(0, 0.7, n)
    mu -= mu.mean()
    D = 0.5 + rng.random(n)
    J = top_k_jacobian(mu, k, D=D)
    h = 1e-5
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        fd = (top_k_probabilities(mu + e, k, D=D)
              - top_k_probabilities(mu - e, k, D=D)) / (2 * h)
        assert np.abs(J[:, j] - fd).max() < 1e-8
    off = J - np.diag(np.diag(J))
    assert np.abs(off - off.T).max() < 1e-14
    assert np.abs(J.sum(axis=0)).max() < 1e-14
    assert np.linalg.eigvalsh(-(J + J.T) / 2).min() > -1e-12
    J1 = top_k_jacobian(mu, 1, D=D)
    assert np.abs(J1 - race_jacobian(mu, D=D)).max() < 1e-12


def test_sigma_jacobian_and_euler_identity():
    """The analytic sigma derivatives: off-diagonals are the mu pair
    integrand weighted by the standardized coordinate, the own term
    rides the forward pass's membership factor. Verified against finite
    differences (3.4e-10 when pinned) and by the scale-gauge Euler
    identity mu . J_mu + sigma . J_sigma = 0, which held at 5.6e-17."""
    from winning.factor.topk import (top_k_jacobians, top_k_jacobian,
                                     top_k_probabilities)

    rng = np.random.default_rng(2)
    n, k = 6, 2
    mu = rng.normal(0, 0.7, n)
    mu -= mu.mean()
    sd = np.exp(rng.normal(0, 0.25, n))
    D = sd ** 2
    Jm, Js = top_k_jacobians(mu, k, D=D)
    assert np.abs(Jm - top_k_jacobian(mu, k, D=D)).max() < 1e-14
    h = 1e-6
    for j in range(n):
        sdp = sd.copy()
        sdp[j] += h
        sdm = sd.copy()
        sdm[j] -= h
        fd = (top_k_probabilities(mu, k, D=sdp ** 2)
              - top_k_probabilities(mu, k, D=sdm ** 2)) / (2 * h)
        assert np.abs(Js[:, j] - fd).max() < 1e-7
    assert np.abs(Jm @ mu + Js @ sd).max() < 1e-13


def test_bases_against_monte_carlo():
    """Density-agnosticism of the count machinery: gumbel, logistic and
    laplace top-k against argpartition Monte Carlo, one field each."""
    from winning.factor.topk import top_k_probabilities

    rng = np.random.default_rng(9)
    n, k = 9, 3
    mu = rng.normal(0, 0.6, n)
    mu -= mu.mean()
    D = 0.6 + 0.5 * rng.random(n)
    sd = np.sqrt(D)
    M = 400_000
    euler_gamma = 0.5772156649015329
    draws = {
        # the engine's min-gumbel base: eps = (gamma - G) sqrt(6)/pi,
        # zero mean, unit variance, left-skewed
        "gumbel": (euler_gamma - rng.gumbel(0, 1, (M, n)))
        * np.sqrt(6) / np.pi,
        "logistic": rng.logistic(0, 1, (M, n)) * np.sqrt(3) / np.pi,
        "laplace": rng.laplace(0, 1 / np.sqrt(2), (M, n)),
    }
    for base, eps in draws.items():
        q = top_k_probabilities(mu, k, D=D, base=base)
        X = mu[None, :] + sd[None, :] * eps
        thresh = np.partition(X, k - 1, axis=1)[:, k - 1]
        emp = (X <= thresh[:, None]).mean(axis=0)
        tv = 0.5 * np.abs(q - emp).sum()
        assert abs(q.sum() - k) < 1e-9, base
        assert tv < 8e-3, (base, tv)


def test_lattice_convergence():
    """Doubling the lattice twice moves the answer at rounding scale:
    the count quadrature is converged at the default."""
    from winning.factor.topk import top_k_probabilities

    rng = np.random.default_rng(1)
    n, k = 25, 8
    mu = rng.normal(0, 1.0, n)
    D = 0.4 + rng.random(n)
    q1 = top_k_probabilities(mu, k, D=D, points=513)
    q2 = top_k_probabilities(mu, k, D=D, points=2049)
    assert np.abs(q1 - q2).max() < 1e-9


def test_complement_jacobian_identity():
    """P(worst k) = 1 - q^(n-k) forces dbottom/dmu = -dq^(n-k)/dmu; the
    top-(n-1) row must therefore be minus the negated race's win-row up
    to reordering. Checked through finite differences of bottom_k."""
    from winning.factor.topk import (bottom_k_probabilities,
                                     top_k_jacobian)

    rng = np.random.default_rng(6)
    n = 7
    mu = rng.normal(0, 0.8, n)
    D = 0.5 + rng.random(n)
    J = top_k_jacobian(mu, n - 2, D=D)
    h = 1e-5
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        fd = (bottom_k_probabilities(mu + e, 2, D=D)
              - bottom_k_probabilities(mu - e, 2, D=D)) / (2 * h)
        assert np.abs(-J[:, j] - fd).max() < 1e-8


def test_ties_and_extreme_spread():
    """Exact duplicates split their membership equally, and a
    twenty-standard-deviation favourite has top-k membership
    indistinguishable from one without tripping the slot check."""
    from winning.factor.topk import top_k_probabilities

    mu = np.array([0.3, 0.3, -0.1, 0.5, 0.5])
    q = top_k_probabilities(mu, 2, D=np.ones(5))
    assert abs(q[0] - q[1]) < 1e-12
    assert abs(q[3] - q[4]) < 1e-12
    mu2 = np.array([-20.0, 0.0, 0.1, 0.2, 0.4, 0.6])
    q2 = top_k_probabilities(mu2, 2, D=np.ones(6))
    assert q2[0] > 1 - 1e-9
    assert abs(q2.sum() - 2) < 1e-9


def test_euler_identity_across_k():
    """The scale gauge holds row by row for every k: a joint rescaling
    of locations and spreads moves no membership."""
    from winning.factor.topk import top_k_jacobians

    rng = np.random.default_rng(12)
    n = 6
    mu = rng.normal(0, 0.7, n)
    sd = np.exp(rng.normal(0, 0.3, n))
    for k in (1, 2, 4, 5):
        Jm, Js = top_k_jacobians(mu, k, D=sd ** 2)
        assert np.abs(Jm @ mu + Js @ sd).max() < 1e-12, k
