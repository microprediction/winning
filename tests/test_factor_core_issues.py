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


# ---------------------------------------------------------------- #99

def test_topk_checks_the_invariants_it_returns_99():
    from winning.factor.topk import _checked_topk
    with pytest.raises(RuntimeError):
        _checked_topk(np.array([0.465219, 1.529295, 0.0002206]), 2, "top-k")
    q = top_k_probabilities(np.array([9.23877796, -4.9945676, 6.54831504]),
                            2, D=np.array([998.259121, 0.0465499142,
                                           0.00819385198]), points=513)
    assert abs(q.sum() - 2) < 1e-9 and q.max() <= 1.0
    assert abs(q[2] - 0.533930405) < 1e-6


# ---------------------------------------------------------------- #340

def test_topk_factor_nodes_adapt_to_sharpness_340():
    mu = np.array([-0.15, 0.05, 0.10, 0.0])
    V = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    D = np.full(4, 0.01)
    race = race_probabilities(mu, V=V, D=D, points=1025)
    top1 = top_k_probabilities(mu, 1, V=V, D=D, points=1025)
    top2 = top_k_probabilities(mu, 2, V=V, D=D, points=1025)
    R = rank_probabilities(mu, V=V, D=D, points=1025)
    assert np.abs(top1 - race).max() < 1e-6
    sobol = np.array([0.52266979, 0.49723697, 0.47734331, 0.50275040])
    assert np.abs(top2 - sobol).max() < 2e-4          # 3.8e-5 MC noise
    assert np.abs(R[:, :2].sum(axis=1) - top2).max() < 1e-6


def test_topk_rank_two_is_rotation_invariant_340():
    mu = np.array([1.1044040028798148, -0.610358146270428,
                   -0.7073887878365588, 0.7951985223407908])
    D = np.array([0.09300996937789023, 0.30829572594491766,
                  0.08224545726086945, 0.25876414192141967])
    V = np.array([[-0.5506448387040641, 0.9898350311233367],
                  [0.04244734092688795, -0.0540614843414359],
                  [0.112700509026512, -1.086117151256921],
                  [0.5576774312714896, 1.1493443752646453]])
    th = 0.47
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = top_k_probabilities(mu, 2, V=V, D=D)
    q = top_k_probabilities(mu, 2, V=V @ Q, D=D)
    assert np.abs(p - q).max() < 1e-3                 # was 1.3e-2


# ---------------------------------------------------------------- #365

@pytest.mark.parametrize("points", [513, 2049, 8193])
def test_bottom_k_keeps_rare_tails_365(points):
    from winning.factor.topk import bottom_k_probabilities
    mu = np.array([-12.0, 0.0, 0.0])
    b = bottom_k_probabilities(mu, 1, D=np.ones(3), points=points)
    assert abs(b[0] / 7.726967753938468796e-24 - 1) < 1e-6
    R = rank_probabilities(mu, D=np.ones(3), points=points)
    assert abs(b[0] / R[0, 2] - 1) < 1e-6


def test_bottom_k_agrees_with_rank_columns_365():
    from winning.factor.topk import bottom_k_probabilities
    rng = np.random.default_rng(1)
    n = 6
    mu = rng.normal(size=n)
    D = rng.uniform(0.3, 2.0, n)
    for k in (1, 2, 4):
        b = bottom_k_probabilities(mu, k, D=D)
        R = rank_probabilities(mu, D=D)[:, n - k:].sum(axis=1)
        assert np.abs(b - R).max() < 1e-7
        assert np.abs(b - (1 - top_k_probabilities(mu, n - k, D=D))).max() \
            < 1e-12


# ---------------------------------------------------------------- #317

@pytest.mark.parametrize("r", [1.5, 1.999, 2.0001, 1 + 1e-9, np.nan, np.inf,
                               "1.5"])
def test_fractional_ranks_are_refused_317(r):
    from winning.factor.topk import abilities_from_rank_marginal
    mu = np.array([-0.7, -0.1, 0.25, 0.55])
    D = np.array([0.7, 0.8, 0.9, 1.0])
    P = rank_probabilities(mu, D=D, points=1025)
    with pytest.raises(ValueError):
        abilities_from_rank_marginal(P[:, 0], r, D=D, points=1025, mu0=mu)


@pytest.mark.parametrize("r", [2, 2.0, np.int64(2)])
def test_whole_number_ranks_are_accepted_317(r):
    from winning.factor.topk import abilities_from_rank_marginal
    mu = np.array([-0.7, -0.1, 0.25, 0.55])
    D = np.array([0.7, 0.8, 0.9, 1.0])
    P = rank_probabilities(mu, D=D, points=1025)
    _, info = abilities_from_rank_marginal(P[:, 1], r, D=D, points=1025,
                                           mu0=mu, return_info=True)
    assert info["converged"]


# ---------------------------------------------------------------- #378

def test_middle_rank_needs_a_branch_378():
    from winning.factor.topk import abilities_from_rank_marginal
    mu = np.array([-1, -0.4, 0.05, 0.45, 0.9])
    D = np.ones(5)
    t = rank_probabilities(mu, D=D, points=257)[:, 2]
    with pytest.raises(ValueError, match="mu0"):
        abilities_from_rank_marginal(t, 3, D=D, points=257)
    _, info = abilities_from_rank_marginal(t, 3, D=D, points=257, mu0=mu,
                                           return_info=True)
    assert info["converged"]
    # not the middle rank, or an asymmetric base: the zero start is fine
    abilities_from_rank_marginal(rank_probabilities(mu, D=D)[:, 1], 2, D=D,
                                 return_info=True)


# ---------------------------------------------------------------- #360

def test_exact_warm_start_returns_the_canonical_gauge_360():
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     loc_scale_from_win_and_second)
    mu = np.array([-3.0, -1.0, 1.0, 3.0])
    sd = np.array([2.0, 3.0, 4.0, 5.0])
    D = sd ** 2
    q1 = top_k_probabilities(mu, 1, D=D, points=1025)
    q2 = top_k_probabilities(mu, 2, D=D, points=1025)
    wm, ws, wi = loc_scale_from_topk_pair(q1, 1, q2, 2, D0=D, mu0=mu,
                                          points=1025, return_info=True)
    cm, cs, ci = loc_scale_from_topk_pair(q1, 1, q2, 2, points=1025,
                                          return_info=True)
    assert wi["converged"] and ci["converged"]
    assert abs(np.mean(wm)) < 1e-12
    assert abs(np.exp(np.mean(np.log(ws))) - 1) < 1e-12
    assert np.abs(wm - cm).max() < 1e-6 and np.abs(ws - cs).max() < 1e-6
    R = rank_probabilities(mu, D=D, points=1025)
    am, as_, _ = loc_scale_from_win_and_second(R[:, 0], R[:, 1], D0=D,
                                               mu0=mu, points=1025,
                                               return_info=True)
    assert abs(np.exp(np.mean(np.log(as_))) - 1) < 1e-12


# ---------------------------------------------------------------- #353 #105

@pytest.mark.parametrize("ridge", [0.0, 1e-4, 0.0025])
def test_a_stall_on_an_impossible_board_is_not_convergence_353(ridge):
    from winning.factor.topk import loc_scale_from_topk_pair
    for q1, q2, pts in ((np.array([0.8, 0.1, 0.1]),
                         np.array([0.2, 0.9, 0.9]), 1025),
                        (np.array([0.9, 0.04, 0.03, 0.03]),
                         np.array([0.1, 0.7, 0.6, 0.6]), 65)):
        _, _, info = loc_scale_from_topk_pair(q1, 1, q2, 2, points=pts,
                                              ridge=ridge, return_info=True)
        assert not info["converged"]
        assert not info["nested"] and not info["fit_converged"]


def test_a_ridge_optimum_on_a_feasible_board_still_converges_353():
    from winning.factor.topk import loc_scale_from_topk_pair
    rng = np.random.default_rng(41)
    n = 8
    mu = rng.normal(size=n) * 0.7
    sd = np.exp(rng.uniform(-0.25, 0.25, size=n))
    q1 = top_k_probabilities(mu, 1, D=sd ** 2)
    q2 = top_k_probabilities(mu, 2, D=sd ** 2)
    _, _, info = loc_scale_from_topk_pair(q1, 1, q2, 2, ridge=0.05,
                                          return_info=True)
    assert info["converged"] and info["stationary"] and info["nested"]
    assert not info["fit_converged"]


# ---------------------------------------------------------------- #104

def test_polish_reaches_the_feasible_pair_104():
    from scipy.special import ndtri
    from winning.factor.polish import polish_race
    mu = np.array([-8 / np.sqrt(2), 8 / np.sqrt(2)])
    p, m, info = polish_race(mu0=mu, D=np.ones(2),
                             name_caps=np.array([0.9, np.nan]))
    assert info["converged"] and info["max_violation"] < 1e-9
    assert abs(p[0] - 0.9) < 1e-6
    assert np.abs(m - np.array([-1, 1]) * ndtri(0.9) / np.sqrt(2)).max() \
        < 1e-5


def test_polish_reports_infeasible_constraints_104():
    from winning.factor.polish import polish_race
    with pytest.warns(RuntimeWarning, match="feasible"):
        _, _, info = polish_race(mu0=np.array([-1.0, 0.0, 1.0]),
                                 D=np.ones(3),
                                 name_caps=np.array([0.2, 0.2, 0.2]))
    assert not info["converged"] and info["max_violation"] > 0.1


# ---------------------------------------------------------------- #430

def _linkage_fixture():
    import json
    from pathlib import Path
    path = Path(__file__).parent / "golden" / "linkage_leaf_variance.json"
    return json.loads(path.read_text())


def test_from_linkage_keeps_near_duplicate_leaf_variances_430():
    fx = _linkage_fixture()
    for case in fx["cases"]:
        t = Tree.from_linkage(np.array(case["Z"], float))
        np.testing.assert_allclose(t.D, case["D"], rtol=fx["rtol"], atol=0,
                                   err_msg=case["name"])
        np.testing.assert_allclose(np.asarray(t.strength)[len(t.D):],
                                   case["strength"][len(t.D):],
                                   rtol=fx["rtol"], atol=0,
                                   err_msg=case["name"])


@pytest.mark.parametrize("h", [1e-5, 1e-6, 1e-8])
def test_near_duplicate_pair_prices_the_cophenetic_model_430(h):
    t = Tree.from_linkage(np.array([[0, 1, h, 2]]))
    # the root shock is common to both leaves and cancels, so the pair is
    # the residual race alone: Phi(1) at every h
    p = race_probabilities(np.array([-h, h]), D=t.D)
    assert abs(p[0] - norm.cdf(1.0)) < 1e-9


def test_zero_height_merge_is_refused_430():
    with pytest.raises(ValueError, match="height 0"):
        Tree.from_linkage(np.array(_linkage_fixture()["refuse_zero_height"]))


# ---------------------------------------------------------------- #350

@pytest.mark.parametrize("root", [0.0, 0.8, 2.0])
def test_tree_jacobian_is_common_root_invariant_350(root):
    from winning.factor.blocks import tree_race_jacobian
    mu = np.array([-0.4, 0.4])
    J = tree_race_jacobian(mu, np.array([0, 1]), np.array([0.3, 0.4]),
                           np.array([0.7, 0.9]), np.array([2, 2, -1]),
                           np.array([0.0, 0.0, root]))
    exact = norm.pdf(0.8 / np.sqrt(1.85)) / np.sqrt(1.85)
    assert abs(J[0, 1] - exact) < 1e-8
    assert abs(J[0, 0] + exact) < 1e-8


def test_tree_jacobian_is_the_forward_derivative_350():
    """A deeper tree: siblings 0, 1 share a non-root ancestor."""
    from winning.factor.blocks import (tree_race_jacobian,
                                       tree_race_probabilities)
    from winning.factor.polish import _structure_engines
    mu = np.array([-0.5, -0.1, 0.2, 0.4])
    args = (np.array([0, 1, 2, 2]), np.array([0.2, 0.3, -0.1, 0.4]),
            np.array([0.8, 0.9, 1.0, 1.1]), np.array([3, 3, 4, 4, -1]),
            np.array([0.0, 0.0, 0.0, 0.6, 0.4]))
    J = tree_race_jacobian(mu, *args)
    h = 1e-4
    for j in range(4):
        e = np.zeros(4); e[j] = h
        fd = (tree_race_probabilities(mu + e, *args)
              - tree_race_probabilities(mu - e, *args)) / (2 * h)
        assert np.abs(J[:, j] - fd).max() < 1e-7
    _, jac, _ = _structure_engines(Tree(*args), 257)     # polish's gradient
    assert np.abs(jac(mu) - J).max() < 1e-12
