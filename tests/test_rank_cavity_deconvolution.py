"""Rare exact places must not be deconvolution residue (#511).

The leave-one-out cavity pmf is recovered from the shared count
distribution by deconvolution. The old rule ran one direction per lattice
point; that bounds error growth but not the cancellation in each
subtraction, so a rare coefficient came back as roundoff residue and was
integrated as real exact-place mass: 62x the true last place in a Gumbel
field, 1.6e12x for a normal one, and a top-1 share 1.8e5x too large."""
import itertools
import warnings

import numpy as np
import pytest

import winning.factor.topk as T
from winning.factor.races import BASES, race_probabilities
from winning.factor.topk import rank_probabilities, top_k_probabilities


def _plackett_luce_ranks(mu):
    """Independent standardised min-Gumbels induce Plackett-Luce with
    weights exp(-pi/sqrt(6) mu); enumerate every order."""
    w = np.exp(-np.pi / np.sqrt(6.0) * np.asarray(mu, float))
    n = len(w)
    P = np.zeros((n, n))
    for order in itertools.permutations(range(n)):
        p = 1.0
        for r in range(n - 1):
            p *= w[order[r]] / w[list(order[r:])].sum()
        for r, i in enumerate(order):
            P[i, r] += p
    return P


@pytest.mark.parametrize("points", [129, 513, 8193])
def test_gumbel_rank_matrix_matches_plackett_luce_entrywise(points):
    mu = [-9.0, -2.0, 2.0, 10.0]
    exact = _plackett_luce_ranks(mu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = rank_probabilities(np.array(mu), D=np.ones(4), base="gumbel",
                               points=points)
    # relative, entry by entry: the old matrix was doubly stochastic and
    # still put 9.2e-19 on a 1.48e-20 last place
    assert exact[0, 3] == pytest.approx(1.4753035919935722e-20, rel=1e-12)
    assert np.max(np.abs(P / exact - 1.0)) < 1e-5


@pytest.mark.parametrize("points", [513, 8193])
def test_normal_last_place_is_the_cavity_product(points):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = rank_probabilities(np.array([-12.0, -4.0, 4.0, 12.0]),
                               points=points)
    assert P[1, 3] == pytest.approx(2.9183334444853456e-30, rel=1e-9)


@pytest.mark.parametrize("points", [513, 8193])
def test_top1_equals_the_win_market_in_the_tail(points):
    mu = np.array([-14.0, 0.0, 0.0])
    q = top_k_probabilities(mu, 1, points=points)
    assert q[1] == pytest.approx(2.091912795347403e-23, rel=1e-9)
    assert q[1] == pytest.approx(race_probabilities(mu, points=points)[1],
                                 rel=3e-3)


def _field(base, mu, sd, points=1025):
    br = BASES[base]
    x = np.linspace(mu.min() - 10 * sd.max(), mu.max() + 10 * sd.max(),
                    points)
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = br(z)
    F, S = T._cdf_rows(br, z, S)
    return F, S, f / sd[None, :], x[1] - x[0]


@pytest.mark.parametrize("seed", range(12))
def test_every_cavity_integral_matches_a_direct_product(seed):
    """Each runner's exact-place and top-k integrals from the deconvolved
    cavity against the count program run directly without that runner,
    on the same lattice -- relative, down to 1e-100."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    mu = rng.normal(0.0, [1.0, 4.0, 10.0][seed % 3], n)
    sd = np.exp(rng.normal(0.0, 0.3, n))
    base = ["normal", "gumbel", "logistic"][seed % 3]
    F, S, dens, dx = _field(base, mu, sd)
    C = T._count_distribution(F, S)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        direct = T._count_distribution(F[:, keep], S[:, keep])
        d = dens[:, i][:, None]
        got = (T._loo_pmf(C, F, i, S=S) * d).sum(axis=0) * dx
        want = (direct * d).sum(axis=0) * dx
        ok = want > 1e-100
        assert np.all(np.abs(got[ok] / want[ok] - 1.0) < 1e-9)
        for k in range(1, n):
            cdf = T._leave_one_out_cdf(C, F, k, S=S)[i]
            g = (cdf * dens[:, i]).sum() * dx
            w = (direct[:, :k].sum(axis=1) * dens[:, i]).sum() * dx
            if w > 1e-100:
                # accepted below a bound of 1e-6 of the value per point
                assert abs(g / w - 1.0) < 1e-6


def test_the_reported_bound_covers_the_actual_error():
    rng = np.random.default_rng(3)
    mu = rng.normal(0.0, 8.0, 7)
    sd = np.exp(rng.normal(0.0, 0.3, 7))
    F, S, _, _ = _field("gumbel", mu, sd)
    C = T._count_distribution(F, S)
    E = T._count_error(C)
    for i in range(7):
        keep = [j for j in range(7) if j != i]
        direct = T._count_distribution(F[:, keep], S[:, keep])
        for fwd in (True, False):
            for m, q, e in T._deconv(C, E, F[:, i], S[:, i], fwd):
                assert np.all(np.abs(q - direct[:, m]) <= e + 1e-300)


def test_a_dead_divisor_carries_no_information():
    """F_i == 0 makes C blind to the coefficients the backward pass would
    divide out; its bound must say so rather than report them as exact."""
    F = np.array([[0.0, 2.5e-5, 0.0]])
    S = 1.0 - F
    C = T._count_distribution(F, S)
    cdf = T._leave_one_out_cdf(C, F, 1, S=S)
    assert cdf[0, 0] == pytest.approx(1.0 - 2.5e-5, rel=1e-14)
