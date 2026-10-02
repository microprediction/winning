"""Regressions for the factor-core issue batch (races, top-k, inverse,
structure dispatch). Each test names its issue; each was run against the
pre-fix tree and failed there."""
import warnings

import numpy as np
import pytest
from scipy.stats import norm

import winning
from winning.factor import race_probabilities
from winning.factor.core import (abilities_from_probabilities_factor,
                                 as_loadings, hermite_nodes,
                                 jacobian_vector_product,
                                 win_probabilities_factor)
from winning.factor.permutations import ordered_probabilities
from winning.factor.races import abilities_from_race, removal_shares
from winning.factor.structures import Blocks, Factor, Independent, Nested, Tree
from winning.factor.topk import (abilities_from_topk, rank_probabilities,
                                 top_k_probabilities)
from winning.shapes import as_weights


# ---------------------------------------------------------------- #263

def test_weights_normalise_without_overflow_263():
    """[1e308, 1e308] is the same law as [1, 1]; the sum overflowed."""
    assert np.allclose(as_weights([1e308, 1e308]), [0.5, 0.5])
    mu = np.array([0.0, 0.3, 1.0])
    V = np.array([[0.0], [1.0], [-0.5]])
    F = np.array([[-1.0], [1.0]])
    for fn, kw in ((race_probabilities, {"D": np.ones(3)}),
                   (winning.softmax_probabilities, {})):
        a = np.asarray(fn(mu, V=V, F=F, W=np.array([1e307, 1e307]), **kw))
        b = np.asarray(fn(mu, V=V, F=F, W=np.array([1e308, 1e308]), **kw))
        assert np.isfinite(b).all() and np.abs(a - b).max() < 1e-12
    la = winning.plackett_luce_order_logprob(mu, [0, 1], V=V, F=F,
                                             W=np.array([1e307, 1e307]))
    lb = winning.plackett_luce_order_logprob(mu, [0, 1], V=V, F=F,
                                             W=np.array([1e308, 1e308]))
    assert np.isfinite(lb) and abs(la - lb) < 1e-12


def test_ordered_logprob_refuses_a_signed_rule_263():
    mu = np.array([0.0, 0.3, 1.0])
    V = np.array([[0.0], [1.0], [-0.5]])
    with pytest.raises(ValueError, match="negative"):
        winning.plackett_luce_order_logprob(
            mu, [0, 1], V=V, F=np.array([[-5.0], [5.0]]),
            W=np.array([2.0, -1.0]))


# ---------------------------------------------------------------- #416

def _zero_padded():
    mu = np.array([0.1, 0.2, 0.4])
    V = np.array([[2.0], [0.0], [-2.0]])
    D = np.full(3, 1e-6)
    return mu, V, D


def test_zero_weight_nodes_are_no_ops_in_the_core_416():
    mu, V, D = _zero_padded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centre = win_probabilities_factor(mu, V, D, np.array([[0.0]]),
                                          np.array([1.0]), points=501)
        padded = win_probabilities_factor(
            mu, V, D, np.array([[-100.0], [0.0], [100.0]]),
            np.array([0.0, 1.0, 0.0]), points=501)
        front = race_probabilities(
            mu, V=V, D=D, F=np.array([[-100.0], [0.0], [100.0]]),
            W=np.array([0.0, 1.0, 0.0]), points=501)
    assert padded[0] > 0.999 and centre[0] > 0.999
    assert np.abs(padded - centre).max() < 1e-9
    assert np.abs(front - centre).max() < 1e-9


def test_core_fallback_keeps_the_callers_factor_law_416(monkeypatch):
    """Force the span-window failure: the retry must price (F, W), not the
    default Gaussian-factor law, and report the failed total."""
    import winning.factor.core as core
    mu = np.array([0.0, 0.4, 0.9])
    V = np.array([[0.5], [0.0], [-0.5]])
    D = np.ones(3)
    F = np.array([[-1.5], [1.5]])
    W = np.array([0.8, 0.2])
    want = race_probabilities(mu, V=V, D=D, F=F, W=W, points=257)
    default = race_probabilities(mu, V=V, D=D, points=257)
    assert np.abs(want - default).max() > 1e-2       # the laws differ
    monkeypatch.setattr(core, "_HAVE_RUST", False)
    real_log_ndtr = core.log_ndtr
    monkeypatch.setattr(core, "log_ndtr", lambda z: real_log_ndtr(z) - 1e4)
    p, total = core.win_probabilities_factor(mu, V, D, F, W, points=501,
                                             return_total=True)
    assert total == 0.0
    assert np.abs(p - want).max() < 1e-12


# ---------------------------------------------------------------- #444

@pytest.mark.parametrize("points", [0, 1, 2.5, -3, np.nan])
def test_lattice_size_must_be_at_least_two_444(points):
    mu = np.array([-0.5, 0.0, 0.5])
    V = np.array([[-0.2], [0.0], [0.2]])
    D = np.ones(3)
    F = np.array([[-1.0], [1.0]])
    W = np.array([0.5, 0.5])
    calls = [
        lambda: win_probabilities_factor(mu, V, D, F, W, points=points),
        lambda: race_probabilities(mu, V=V, D=D, points=points),
        lambda: abilities_from_race([0.5, 0.3, 0.2], D=D, points=points),
        lambda: removal_shares(mu, D=D, points=points),
        lambda: ordered_probabilities(mu, k=2, D=D, points=points),
        lambda: abilities_from_probabilities_factor(
            [0.5, 0.3, 0.2], V, D, F, W, points=points),
        lambda: jacobian_vector_product(mu, V, D, F, W, np.ones(3),
                                        points=points),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="points"):
            call()


def test_two_points_is_a_legal_lattice_444():
    mu = np.array([-0.5, 0.0, 0.5])
    p = race_probabilities(mu, D=np.ones(3), points=2, window="span")
    assert np.isfinite(p).all()


# ---------------------------------------------------------------- #110

@pytest.mark.parametrize("bad", [[np.nan, 0.5], [np.inf, 1.0],
                                 [0.8, np.nan], [-np.inf, 1.0]])
def test_non_finite_targets_are_refused_110(bad):
    F, W = hermite_nodes(1)
    with pytest.raises(ValueError, match="finite"):
        abilities_from_probabilities_factor(bad, V=[0, 0], D=[1, 1], F=F,
                                            W=W, return_info=True)
    with pytest.raises(ValueError, match="finite"):
        abilities_from_race(bad, return_info=True)


def test_non_vector_targets_are_refused_110():
    with pytest.raises(ValueError):
        abilities_from_race([[0.5, 0.5]], return_info=True)


@pytest.mark.parametrize("target", [[1.0, 1e-17], [0.9999999999999999, 1e-20],
                                    [1e20, 1.0], [1e-17, 1.0]])
def test_pair_closed_forms_use_the_longshot_tail_110(target):
    t = np.asarray(target, float)
    small = int(np.argmin(t))
    want = t[small] / t.sum()
    mu, info = abilities_from_race(t, D=[1, 1], return_info=True)
    assert np.isfinite(mu).all() and info["converged"]
    p = race_probabilities(mu, D=[1, 1])
    assert abs(np.log(p[small]) - np.log(want)) < 1e-8
    F, W = hermite_nodes(1)
    mu2, info2 = abilities_from_probabilities_factor(
        t, V=[0, 0], D=[1, 1], F=F, W=W, return_info=True)
    assert np.isfinite(mu2).all()
    assert np.allclose(mu2, mu, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- #70

def test_core_inverse_is_invariant_to_common_loading_shifts_70():
    mu0 = np.array([-0.8, -0.3, 0.0, 0.4, 0.7])
    n = len(mu0)
    D = np.ones(n)
    p = race_probabilities(mu0, V=None, D=D, points=1001)
    F, W = hermite_nodes(1, Q=15)
    V = np.array([0.2, -0.1, 0.3, 0.0, -0.4])
    out = []
    for c in (0.0, 100.0):
        m, info = abilities_from_probabilities_factor(
            race_probabilities(mu0, V=V, D=D, points=1001), V + c, D, F, W,
            n_iter=50, tol=1e-7, return_info=True, points=501)
        assert info["converged"]
        out.append(m - m.mean())
    assert np.abs(out[0] - out[1]).max() < 1e-6
    m, info = abilities_from_probabilities_factor(
        p, 100.0, D, F, W, n_iter=50, tol=1e-7, return_info=True, points=501)
    assert info["converged"]
    assert np.abs((m - m.mean()) - (mu0 - mu0.mean())).max() < 1e-5


# ---------------------------------------------------------------- #71

def test_square_loadings_are_read_as_the_contract_shape_71():
    """No shape-only rule can tell (n, n) from its transpose; the contract
    wins, so V.T is a DIFFERENT race (V'V, not VV'), as documented."""
    rng = np.random.default_rng(7)
    n = 4
    V = rng.normal(size=(n, n))
    assert np.array_equal(as_loadings(V.T, n), V.T)
    mu = np.array([-0.5, -0.2, 0.1, 0.6])
    D = np.full(n, 0.7)
    p = race_probabilities(mu, V=V, D=D, points=513)
    pt = race_probabilities(mu, V=V.T, D=D, points=513)
    assert np.abs(p - pt).max() > 1e-2
    assert "rank == n" in as_loadings.__doc__
    assert "square" in race_probabilities.__doc__


# ---------------------------------------------------------------- #100

@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_topk_inverse_is_unit_equivariant_100(c):
    mu0 = np.array([-0.8, -0.3, 0.0, 0.4, 0.7])
    n = len(mu0)
    q = top_k_probabilities(mu0, 2, D=np.ones(n))
    m, info = abilities_from_topk(q, 2, D=c * c * np.ones(n),
                                  return_info=True)
    assert info["converged"]
    assert np.abs(m / c - (mu0 - mu0.mean())).max() < 1e-6


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_block_and_structure_inverses_are_unit_equivariant_100(c):
    from winning.factor.blocks import (abilities_from_block_race,
                                       block_race_probabilities)
    target = np.array([0.6, 0.25, 0.1, 0.05])
    cl = np.array([0, 0, 1, 1])
    L0 = np.array([0.9, 0.6, 0.8, 0.3])
    D0 = np.array([0.19, 0.64, 0.36, 0.91])
    mu, res, _ = abilities_from_block_race(target, cl, c * L0, c * c * D0)
    back = block_race_probabilities(mu, cl, c * L0, c * c * D0)
    assert np.abs(back - target).max() < 1e-9
    s = Nested(cluster=cl, loading=c * L0, D=c * c * D0,
               coupling=c * np.array([[0.2], [-0.1], [0.3], [-0.2]]),
               gamma=0.7)
    mu, info = abilities_from_race(target, structure=s, return_info=True)
    assert info["converged"]
    assert np.abs(race_probabilities(mu, structure=s) - target).max() < 1e-8


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_core_inverse_is_unit_equivariant_100(c):
    mu0 = np.array([-0.8, -0.3, 0.0, 0.4, 0.7])
    n = len(mu0)
    p = race_probabilities(mu0, D=np.ones(n), points=1001)
    F, W = hermite_nodes(1, Q=15)
    m, info = abilities_from_probabilities_factor(
        p, np.zeros((n, 1)), c * c * np.ones(n), F, W, return_info=True,
        n_iter=50, tol=1e-7)
    assert info["converged"]
    assert np.abs(m / c - (mu0 - mu0.mean())).max() < 1e-6


# ---------------------------------------------------------------- #370

@pytest.mark.parametrize("c", [1e-18, 1e-9, 1e9])
def test_topk_window_is_scale_free_370(c):
    mu = np.array([-0.5, 0.2, 0.8, -0.1])
    sd = np.array([0.7, 1.1, 0.9, 1.3])
    race = race_probabilities(mu, D=sd ** 2)
    for k in (1, 2):
        a = top_k_probabilities(mu, k, D=sd ** 2)
        b = top_k_probabilities(c * mu, k, D=(c * sd) ** 2)
        assert np.abs(a - b).max() < 1e-9
    assert np.abs(top_k_probabilities(c * mu, 1, D=(c * sd) ** 2)
                  - race).max() < 1e-6
    R = rank_probabilities(c * mu, D=(c * sd) ** 2)
    assert np.abs(R - rank_probabilities(mu, D=sd ** 2)).max() < 1e-9


# ---------------------------------------------------------------- #411

@pytest.mark.parametrize("base", ["normal", "laplace", "logistic", "gumbel"])
@pytest.mark.parametrize("gap", [0.0, 10.0, 40.0])
def test_two_runner_removal_is_the_permutation_matrix_411(base, gap):
    q = removal_shares(np.array([0.0, gap]), D=np.array([1.0, 0.3]),
                       base=base)
    assert np.array_equal(q, [[0.0, 1.0], [1.0, 0.0]])


def test_one_runner_removal_is_refused_411():
    with pytest.raises(ValueError, match="two runners"):
        removal_shares(np.array([0.0]), D=np.ones(1))


# ---------------------------------------------------------------- #269

def test_capped_removal_lattice_is_refined_not_trusted_269():
    mu = np.array([-4.599118365656239, 0.00014998701879823104,
                   0.00023837447683767292])
    D = np.array([1e-4, 1e-8, 1e-8])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = removal_shares(mu, D=D, points=501)
    exact = norm.cdf((mu[2] - mu[1]) / np.sqrt(D[1] + D[2]))
    assert abs(q[0, 1] - exact) < 1e-3


# ---------------------------------------------------------------- #424

@pytest.mark.parametrize("tau", [-0.7, np.nan, np.inf, -np.inf])
def test_invalid_temperature_is_refused_424(tau):
    mu = np.array([-0.3, 0.2, 0.8])
    with pytest.raises(ValueError, match="temperature"):
        race_probabilities(mu, temperature=tau)
    with pytest.raises(ValueError, match="temperature"):
        ordered_probabilities(mu, k=2, temperature=tau)
    with pytest.raises(ValueError, match="temperature"):
        abilities_from_race([0.5, 0.3, 0.2], temperature=tau)


def test_zero_temperature_is_still_the_hard_race_424():
    mu = np.array([-0.3, 0.2, 0.8])
    assert np.array_equal(race_probabilities(mu, temperature=0.0),
                          race_probabilities(mu))


# ---------------------------------------------------------------- #89

def test_structured_inverse_refuses_a_second_covariance_89():
    s = Independent(np.ones(3))
    with pytest.raises(ValueError, match="structure"):
        abilities_from_race([0.5, 0.3, 0.2], structure=s, V=np.ones(3))
    with pytest.raises(ValueError, match="structure"):
        abilities_from_race([0.5, 0.3, 0.2], structure=s, D=np.ones(3))


def test_structured_inverse_refuses_unsupported_base_89():
    s = Blocks(cluster=np.array([0, 0, 1, 1]),
               loading=np.array([0.3, 0.2, 0.4, 0.1]), D=np.ones(4))
    for kw in ({"base": "gumbel"}, {"base": "laplace"},
               {"temperature": 0.5}):
        with pytest.raises(NotImplementedError):
            abilities_from_race([0.4, 0.3, 0.2, 0.1], structure=s, **kw)


def test_structured_inverse_honours_n_iter_89():
    s = Blocks(cluster=np.array([0, 0, 1, 1]),
               loading=np.array([0.3, 0.2, 0.4, 0.1]), D=np.ones(4))
    _, info = abilities_from_race([0.4, 0.3, 0.2, 0.1], structure=s,
                                  n_iter=1, return_info=True)
    assert info["iterations"] == 1


# ---------------------------------------------------------------- #73

def _structures():
    cl = np.array([0, 0, 1, 1])
    L = np.array([0.3, 0.2, 0.4, 0.1])
    D = np.ones(4)
    return [
        Blocks(cluster=cl, loading=L, D=D),
        Nested(cluster=cl, loading=L, D=D,
               coupling=np.array([[0.2], [-0.1], [0.3], [-0.2]]), gamma=0.7),
        Tree(cluster=cl, loading=L, D=D, parent=np.array([2, 2, -1]),
             strength=np.array([0.0, 0.0, 0.5])),
    ]


@pytest.mark.parametrize("s", _structures(), ids=lambda s: type(s).__name__)
def test_ordered_k1_prices_every_structure_73(s):
    mu = np.array([-0.4, -0.1, 0.2, 0.5])
    np.testing.assert_allclose(
        ordered_probabilities(mu, k=1, structure=s, points=257),
        race_probabilities(mu, structure=s, points=257))
    with pytest.raises(NotImplementedError):
        ordered_probabilities(mu, k=2, structure=s, points=257)
