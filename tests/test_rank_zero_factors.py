"""Rank zero is the independent race, on every path that takes loadings.

The factor grammar documents the containment `Independent = Factor with
empty V`, and `as_loadings` accepts `(n, 0)` with the comment
`empty = indep`. Four public paths could not price that representation
(#68), each for the same reason: the node constructors build the tensor
with a loop or a meshgrid over the rank, and at rank zero that does not
raise -- it falls through to a rule for a DIFFERENT problem, or into a
numpy message about concatenation.

A zero-dimensional Gaussian integral has one empty node of unit mass.
These tests state that, and then state the consequence, which is the
part a user would notice: the answer equals the independent one.
"""
import numpy as np
import pytest

from winning.factor import race_probabilities
from winning.factor.core import hermite_nodes
from winning.factor.structures import Factor


def test_the_node_rule_itself_is_the_empty_product():
    F, W = hermite_nodes(0, 15)
    assert F.shape == (1, 0)
    assert W.shape == (1,) and W[0] == pytest.approx(1.0)


def test_the_race_prices_the_empty_loading_matrix():
    n = 4
    mu = np.arange(n, dtype=float)
    D = np.ones(n)
    want = race_probabilities(mu, V=None, D=D)
    assert np.allclose(race_probabilities(mu, V=np.empty((n, 0)), D=D), want)
    assert np.allclose(
        race_probabilities(mu, structure=Factor(np.empty((n, 0)), D)), want)


def test_the_transposed_empty_spelling_is_the_same_race():
    """(0, n) normalises to (n, 0) at the door, so it must price too."""
    n = 4
    mu = np.arange(n, dtype=float)
    D = np.ones(n)
    assert np.allclose(race_probabilities(mu, V=np.empty((0, n)), D=D),
                       race_probabilities(mu, V=None, D=D))


@pytest.mark.parametrize("module", ["winning.fastmvn", "fastmvn"])
def test_both_fastmvn_copies_price_the_rank_zero_rectangle(module):
    """The two are byte-identical by contract, so fixing one and not the
    other is not possible -- but importing both is what proves the
    standalone package ships the fix, not just the tree it came from."""
    if module == "fastmvn":
        import os
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = os.path.join(root, "python", "fastmvn", "src")
        if not os.path.isdir(src):
            pytest.skip("standalone fastmvn source not in this tree")
        sys.path.insert(0, src)
    mvn_cdf_fast = __import__(module, fromlist=["mvn_cdf_fast"]).mvn_cdf_fast
    got = mvn_cdf_fast(upper=np.zeros(3), V=np.empty((3, 0)), D=np.ones(3))
    assert got == pytest.approx(0.5 ** 3, abs=1e-12)


def test_top_k_and_rank_price_the_empty_loading_matrix():
    """#68 covered the race and the CDF. These dispatch rank zero
    through their OWN node builders, which special-cased rank 1 and sent
    everything else through a rank-2 tensor (#309)."""
    from winning.factor.topk import (bottom_k_probabilities,
                                     rank_probabilities,
                                     top_k_probabilities)
    n = 4
    mu = np.array([0.0, 0.3, -0.2, 0.5])
    D = np.ones(n)
    V0 = np.empty((n, 0))
    assert np.allclose(top_k_probabilities(mu, 2, V=V0, D=D),
                       top_k_probabilities(mu, 2, D=D))
    assert np.allclose(bottom_k_probabilities(mu, 2, V=V0, D=D),
                       bottom_k_probabilities(mu, 2, D=D))
    assert np.allclose(rank_probabilities(mu, V=V0, D=D),
                       rank_probabilities(mu, D=D))


def test_the_ranking_likelihood_takes_the_empty_loading_matrix():
    """`choice_loglik_and_score` already handled it; the ranking member
    raised `need at least one array to concatenate` (#309)."""
    from winning.likelihood import ranking_loglik_and_score
    mu = np.array([[0.0, 0.3, -0.2, 0.5]])
    ll, dmu, dV = ranking_loglik_and_score(mu, np.empty((4, 0)),
                                           [[0, 1, 2, 3]])
    assert np.isfinite(ll)
    assert dmu.shape == (1, 4)
    assert dV.shape == (4, 0)
    # and it equals the plain Plackett-Luce value with no factor
    z = mu[0]
    want = 0.0
    alive = list(range(4))
    for w in [0, 1, 2, 3]:
        want += z[w] - np.log(np.exp(z[alive]).sum())
        alive.remove(w)
    assert ll == pytest.approx(float(want), abs=1e-9)


def test_a_zero_rank_race_is_not_accidentally_the_rank_one_race():
    """The failure mode was a WELL-FORMED rule for another problem, so
    an equality against the independent race only means something if a
    rank-one race is different from it."""
    n = 4
    mu = np.arange(n, dtype=float)
    D = np.ones(n)
    ind = race_probabilities(mu, V=None, D=D)
    one = race_probabilities(mu, V=np.array([[0.9], [-0.4], [0.2], [-0.7]]),
                             D=D)
    assert not np.allclose(ind, one, atol=1e-6)
