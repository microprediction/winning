"""The certified race against planted truths with closed forms, against
the monotone bracket, and against the float engine it certifies."""

import numpy as np
import pytest

flint = pytest.importorskip("flint")
from flint import arb, ctx  # noqa: E402

from winning.certified import (  # noqa: E402
    bracket_race_probabilities,
    certified_race_probabilities,
    contains,
)
from winning.factor import hermite_nodes, race_probabilities  # noqa: E402

DIGITS = 30


@pytest.fixture(autouse=True)
def high_precision():
    # closed forms below are built at this precision
    saved = ctx.prec
    ctx.prec = 200
    yield
    ctx.prec = saved


def _close(ball, truth, digits=DIGITS):
    """Truth inside the ball, and the ball narrow enough to mean it."""
    assert ball.overlaps(truth), (ball.str(35), truth.str(35))
    assert ball.rad() < arb(10) ** (-digits)


def test_two_runners_normal_with_means():
    # P(X1 < X2) = Phi((mu2 - mu1) / sqrt(D1 + D2))
    mu, D = [0.3, -0.4], [1.0, 2.5]
    p = certified_race_probabilities(mu, D)
    z = (arb(mu[1]) - arb(mu[0])) / (arb(D[0]) + arb(D[1])).sqrt()
    truth = (-z / arb(2).sqrt()).erfc() / 2
    _close(p[0], truth)
    _close(p[1], 1 - truth)


def test_three_runners_arcsine():
    # zero means: P(1 wins) = 1/4 + asin(rho)/(2 pi), rho the correlation
    # of (X2 - X1, X3 - X1)
    D = [0.5, 1.0, 2.0]
    p = certified_race_probabilities([0.0, 0.0, 0.0], D)
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        rho = arb(D[i]) / ((arb(D[i]) + D[j]) * (arb(D[i]) + D[k])).sqrt()
        _close(p[i], arb(1) / 4 + rho.asin() / (2 * arb.pi()))


def test_four_runners_orthant():
    # trivariate orthant: 1/8 + (sum of asin rho_jk) / (4 pi)
    D = [0.7, 1.0, 1.3, 2.2]
    p = certified_race_probabilities([0.0] * 4, D)
    for i in range(4):
        others = [m for m in range(4) if m != i]
        s = arb(0)
        for a in range(3):
            for b in range(a + 1, 3):
                j, k = others[a], others[b]
                s += (arb(D[i]) / ((arb(D[i]) + D[j])
                                   * (arb(D[i]) + D[k])).sqrt()).asin()
        _close(p[i], arb(1) / 8 + s / (4 * arb.pi()))


def test_gumbel_is_luce():
    # min-Gumbel with unit variance: p_i proportional to exp(-c mu_i)
    mu = [-0.5, 0.0, 0.2, 0.3, 1.1]
    p = certified_race_probabilities(mu, base="gumbel")
    c = arb.pi() / arb(6).sqrt()
    w = [(-c * arb(m)).exp() for m in mu]
    tot = sum(w[1:], w[0])
    for pi, wi in zip(p, w):
        _close(pi, wi / tot)


@pytest.mark.parametrize("base", ["normal", "gumbel", "logistic", "laplace"])
def test_identical_runners_split_evenly(base):
    p = certified_race_probabilities([0.25] * 3, base=base)
    for pi in p:
        _close(pi, arb(1) / 3)


@pytest.mark.parametrize("base", ["normal", "gumbel", "logistic", "laplace"])
def test_probabilities_sum_to_one(base):
    p = certified_race_probabilities([-0.5, 0.0, 0.2, 0.3], [1, 0.6, 1.4, 1],
                                     base=base)
    _close(sum(p[1:], p[0]), arb(1), digits=DIGITS - 1)


def test_decimal_strings_are_balls():
    # "0.1" is not a float: the answer is a ball over the decimal itself
    p = certified_race_probabilities(["0.1", "0"])
    z = arb("0.1") / arb(2).sqrt()
    truth = (z / arb(2).sqrt()).erfc() / 2
    _close(p[0], truth)


@pytest.mark.parametrize("base", ["normal", "gumbel", "logistic", "laplace"])
def test_bracket_contains_certificate(base):
    mu = [-0.5, 0.0, 0.2, 0.3]
    fine = certified_race_probabilities(mu, base=base)
    coarse = bracket_race_probabilities(mu, base=base, cells=300)
    for b, c in zip(coarse, fine):
        assert b.contains(c)
        assert b.rad() < 0.1


def test_bracket_narrows_first_order():
    mu = [-0.5, 0.0, 0.2, 0.3]
    r1 = bracket_race_probabilities(mu, cells=200)[0].rad()
    r2 = bracket_race_probabilities(mu, cells=800)[0].rad()
    assert 3.0 < float(r1 / r2) < 5.0


@pytest.mark.parametrize("base", ["normal", "gumbel", "logistic", "laplace"])
def test_float_engine_inside_certificate(base):
    # the point of the module: the fast lattice answer, checked. laplace
    # was 4e-4 off here on the uniform lattice before kink_lattice
    rng = np.random.default_rng(7)
    mu = rng.normal(0, 1, 8)
    D = rng.uniform(0.5, 1.5, 8)
    fast = race_probabilities(mu, D=D, base=base)
    sure = certified_race_probabilities(mu, D, base=base, digits=20)
    for f, s in zip(fast, sure):
        assert abs(f - float(s.mid())) < 1e-13


def test_kinked_jacobian_is_the_derivative_of_the_kinked_map():
    from winning.factor.polish import race_jacobian
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 1, 6)
    V = rng.normal(0, 0.5, (6, 1))
    D = rng.uniform(0.5, 1.5, 6)
    J = race_jacobian(mu, V=V, D=D, base="laplace")
    h = 1e-6
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        fd = (race_probabilities(mu + e, V=V, D=D, base="laplace", points=501)
              - race_probabilities(mu - e, V=V, D=D, base="laplace",
                                   points=501)) / (2 * h)
        assert np.abs(J[:, j] - fd).max() < 1e-8


def test_contains_helper():
    p = certified_race_probabilities([0.0, 0.0])
    assert contains(p[0], arb(1) / 2)
    assert not contains(p[0], 0.5 + 1e-12)


def test_refuses_bad_fields():
    with pytest.raises(ValueError):
        certified_race_probabilities([0.0])
    with pytest.raises(ValueError):
        certified_race_probabilities([0.0, 1.0], D=[1.0, 0.0])
    with pytest.raises(ValueError):
        certified_race_probabilities([0.0, 1.0], base="cauchy")


# ---------------------------------------------------------------------------
# the factor race
# ---------------------------------------------------------------------------

from winning.certified import certified_factor_race_probabilities  # noqa: E402

FDIGITS = 16


def _diff_corr(Sigma, i, j, k):
    """Correlation of X_j - X_i and X_k - X_i under Sigma (arb)."""
    c = Sigma[j][k] - Sigma[i][j] - Sigma[i][k] + Sigma[i][i]
    vj = Sigma[j][j] - 2 * Sigma[i][j] + Sigma[i][i]
    vk = Sigma[k][k] - 2 * Sigma[i][k] + Sigma[i][i]
    return c / (vj * vk).sqrt()


def _sigma(V, D):
    n = len(D)
    return [[sum((arb(V[a][m]) * arb(V[b][m]) for m in range(len(V[0]))),
                 arb(0)) + (arb(D[a]) if a == b else 0)
             for b in range(n)] for a in range(n)]


def test_factor_two_runners_with_means():
    mu, V, D = [0.3, -0.4], [[0.8], [-0.5]], [1.0, 2.5]
    p = certified_factor_race_probabilities(mu, V, D, digits=FDIGITS)
    var = arb(D[0]) + arb(D[1]) + (arb(V[0][0]) - arb(V[1][0])) ** 2
    z = (arb(mu[1]) - arb(mu[0])) / var.sqrt()
    truth = (-z / arb(2).sqrt()).erfc() / 2
    _close(p[0], truth, FDIGITS)
    _close(p[1], 1 - truth, FDIGITS)


@pytest.mark.parametrize("V, digits", [
    ([[0.9], [-0.4], [0.2]], FDIGITS),
    # rank two is a three-dimensional cubature: ~16 s at 10 digits
    ([[0.9, 0.1], [-0.4, 0.5], [0.2, -0.7]], 10),
])
def test_factor_three_runners_arcsine(V, digits):
    D = [0.5, 1.0, 2.0]
    p = certified_factor_race_probabilities([0.0] * 3, V, D, digits=digits)
    Sigma = _sigma(V, D)
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        rho = _diff_corr(Sigma, i, j, k)
        _close(p[i], arb(1) / 4 + rho.asin() / (2 * arb.pi()), digits)


def test_factor_four_runners_orthant():
    V, D = [[0.9], [-0.4], [0.2], [0.6]], [0.7, 1.0, 1.3, 2.2]
    p = certified_factor_race_probabilities([0.0] * 4, V, D,
                                            digits=FDIGITS)
    Sigma = _sigma(V, D)
    for i in range(4):
        others = [m for m in range(4) if m != i]
        s = arb(0)
        for a in range(3):
            for b in range(a + 1, 3):
                s += _diff_corr(Sigma, i, others[a], others[b]).asin()
        _close(p[i], arb(1) / 8 + s / (4 * arb.pi()), FDIGITS)


def test_factor_common_loading_is_luce():
    # a loading every runner shares cannot move an argmin
    mu = [-0.5, 0.0, 0.2, 0.3]
    p = certified_factor_race_probabilities(mu, [[0.7]] * 4, base="gumbel",
                                            digits=FDIGITS)
    c = arb.pi() / arb(6).sqrt()
    w = [(-c * arb(m)).exp() for m in mu]
    tot = sum(w[1:], w[0])
    for pi, wi in zip(p, w):
        _close(pi, wi / tot, FDIGITS)


@pytest.mark.parametrize("base", ["normal", "gumbel", "logistic"])
def test_float_factor_engine_converges_to_certificate(base):
    # Measured on this field: the default node rule picks 15 Gauss-Hermite
    # nodes and is 4.9e-7 (normal), 2.2e-5 (logistic) and 3.0e-5 (gumbel)
    # off; the lattice size changes nothing. With 101 nodes the engine
    # lands within 3e-13 of the certified values, so the gap is the
    # factor rule, not the model.
    rng = np.random.default_rng(11)
    mu = rng.normal(0, 1, 5)
    V = rng.normal(0, 0.6, (5, 1))
    D = rng.uniform(0.5, 1.5, 5)
    F, W = hermite_nodes(1, Q=101)
    fast = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base)
    sure = certified_factor_race_probabilities(mu, V, D, base=base,
                                               digits=FDIGITS)
    for f, s in zip(fast, sure):
        assert abs(f - float(s.mid())) < 1e-12


def test_factor_refuses_kinked_base():
    with pytest.raises(NotImplementedError):
        certified_factor_race_probabilities([0.0, 1.0], [[0.5], [0.1]],
                                            base="laplace")
