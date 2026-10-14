"""Contract tests for the two-runner closed form in race_probabilities.

The engine's verify suite exercises its identities, not the contract of a
new code path: the first cut of this closed form passed all 775 checks
while rounding the loser's tail to zero at a 9 sd contrast. These are the
tests that would have caught it.
"""
import numpy as np
import pytest
from scipy.special import ndtr

from winning.factor.races import race_probabilities


# -- closed form ---------------------------------------------------------------

def test_pair_closed_form_keeps_both_gaussian_tails():
    # a 9 sd contrast: the loser is 1.1e-19, which 1 - ndtr(9) rounds to 0
    u = 9.0
    mu = np.array([0.0, u * np.sqrt(2.0)])          # min-wins: runner 0 favourite
    p = race_probabilities(mu, D=np.ones(2))
    assert p[1] > 0.0
    assert p[1] == pytest.approx(ndtr(-u), rel=1e-12)
    assert p[0] == pytest.approx(ndtr(u), rel=1e-15)
    assert np.log(p[1]) == pytest.approx(np.log(ndtr(-u)), abs=1e-9)


def test_pair_closed_form_is_exactly_permutation_equivariant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu = rng.normal(size=2) * 4.0
        D = rng.uniform(0.3, 3.0, size=2)
        V = rng.normal(size=(2, 1))
        p = race_probabilities(mu, V=V, D=D)
        q = race_probabilities(mu[::-1], V=V[::-1], D=D[::-1])
        assert np.array_equal(p, q[::-1])


def test_pair_closed_form_slopes_are_finite_differences():
    mu = np.array([0.3, -0.3]); D = np.array([1.0, 1.5])
    p, sl = race_probabilities(mu, D=D, return_slopes=True)
    eps = 1e-6
    for i in range(2):
        up = mu.copy(); up[i] += eps
        dn = mu.copy(); dn[i] -= eps
        fd = (race_probabilities(up, D=D)[i] - race_probabilities(dn, D=D)[i]) / (2 * eps)
        assert sl[i] == pytest.approx(fd, abs=1e-7)


def test_pair_closed_form_matches_the_lattice():
    # n == 2 never reaches the lattice now, so force it: a third runner
    # placed 10 sd out (min-wins, so it cannot win, p ~ 1e-25) makes a
    # three-runner race whose first two renormalised probabilities are
    # the lattice's answer for the pair. Without loadings the lattice is
    # spectrally accurate (measured <= 1e-13 relative on the loser tail);
    # with rank-1 loadings it runs Gauss-Hermite and sits near 1e-9.
    rng = np.random.default_rng(4)
    for u in (0.5, 2.0, 4.0):
        D = np.array([1.0, 1.5]); sd = np.sqrt(D.sum())
        mu = np.array([0.0, u * sd])
        p2 = race_probabilities(mu, D=D)
        mu3 = np.r_[mu, mu.max() + 10.0 * np.sqrt(D.max())]
        p3 = race_probabilities(mu3, D=np.r_[D, 1.0])
        q = p3[:2] / p3[:2].sum()
        assert q[0] == pytest.approx(p2[0], abs=1e-12)
        assert q[1] == pytest.approx(p2[1], rel=1e-10)
    for _ in range(5):                                   # exercises the -2 Sig01 term
        mu = rng.normal(size=2); D = rng.uniform(0.5, 2.0, size=2)
        V = rng.normal(size=(2, 1)) * 0.7
        p2 = race_probabilities(mu, V=V, D=D)
        far = mu.max() + 10.0 * np.sqrt(D.max() + (V ** 2).max())
        p3 = race_probabilities(np.r_[mu, far], V=np.vstack([V, [[0.0]]]), D=np.r_[D, 1.0])
        q = p3[:2] / p3[:2].sum()
        assert q[0] == pytest.approx(p2[0], abs=1e-7)




def test_every_grammar_prices_a_pair_in_closed_form_86():
    """Blocks, Nested and Tree integrated a lattice for a pair and priced
    a 14-sd tail at 0.107x the exact 1.04e-45; a pair is one Gaussian
    contrast under every grammar."""
    from scipy.special import ndtr
    from winning.factor.blocks import tree_race_probabilities
    from winning.factor.races import race_probabilities
    from winning.factor.structures import Blocks, Nested, Tree
    mu = np.array([0.0, 20.0])
    exact = ndtr(-20 / np.sqrt(2))
    for s in (Blocks(np.array([0, 1]), np.zeros(2), np.ones(2)),
              Nested(np.array([0, 1]), np.zeros(2), np.ones(2), np.zeros(2),
                     0.5),
              Tree(np.array([0, 1]), np.zeros(2), np.ones(2),
                   np.array([2, 2, -1]), np.zeros(3))):
        assert race_probabilities(mu, structure=s)[1] / exact == \
            pytest.approx(1.0, rel=1e-12)
    # correlated pairs: the factor form is the grammar's covariance
    mu = np.array([0.0, 1.3])
    s = Blocks(np.array([0, 0]), np.array([0.8, -0.5]), np.array([1.0, 2.0]))
    u = 1.3 / np.sqrt(1.3 ** 2 + 3.0)
    assert race_probabilities(mu, structure=s)[0] == pytest.approx(ndtr(u))
    s = Nested(np.array([0, 1]), np.array([0.8, -0.5]), np.array([1.0, 2.0]),
               np.array([0.7, 0.2]), 0.6)
    u = 1.3 / np.sqrt(0.64 + 0.25 + 3.0 + (0.6 * 0.5) ** 2)
    assert race_probabilities(mu, structure=s)[0] == pytest.approx(ndtr(u))
    s = Tree(np.array([0, 1]), np.array([0.8, -0.5]), np.array([1.0, 2.0]),
             np.array([2, 3, 3, -1]), np.array([0.0, 0.0, 0.9, 1.5]))
    u = 1.3 / np.sqrt(0.64 + 0.81 + 0.25 + 3.0)
    assert race_probabilities(mu, structure=s)[0] == pytest.approx(ndtr(u))
    ref = tree_race_probabilities(mu, s.leaf_index, s.loading, s.D, s.parent,
                                  s.strength, points=4097, qa=41)
    assert np.abs(race_probabilities(mu, structure=s) - ref).max() < 1e-6
    with pytest.raises(ValueError, match="exactly one root"):
        race_probabilities(mu, structure=Tree(
            np.array([0, 1]), np.zeros(2), np.ones(2), np.array([-1, -1]),
            np.zeros(2)))


def test_pair_jacobian_is_the_closed_form_one_86():
    from scipy.stats import norm
    from winning.factor.polish import race_jacobian
    J = race_jacobian(np.array([0.0, 12 * np.sqrt(2)]), D=np.ones(2))
    d = norm.pdf(12.0) / np.sqrt(2)
    assert J[0, 0] == pytest.approx(-d, rel=1e-10)    # was +1.5e-16
    assert J[0, 1] == pytest.approx(d, rel=1e-10)


@pytest.mark.parametrize("mu,V,want", [
    ([0.0, 1e155], [[1e155], [-1e155]], 0.5),               # [1, 0] before
    ([0.0, 1.4e308], [[7e307, 7e307], [-7e307, -7e307]], 2 ** -0.5),
])
def test_pair_contrast_whose_square_overflows_86(mu, V, want):
    """dV @ dV overflowed for finite loadings, the contrast sd became NaN
    and the 1e-150 fallback priced [1, 0]; in the rank-two case the sd
    itself exceeds the float range while u = 1/sqrt(2) does not."""
    import warnings
    from scipy.special import ndtr
    from winning.factor.races import race_probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = race_probabilities(mu, V=V, D=[1.0, 1.0])
    np.testing.assert_allclose(p, [ndtr(want), ndtr(-want)], rtol=1e-12)


def test_pair_tree_needs_one_leaf_per_runner_86():
    from winning.factor.races import race_probabilities
    from winning.factor.structures import Tree
    common = dict(loading=np.array([1.0, 1.0]), D=np.array([1.0, 1.0]))
    bad = Tree(cluster=[0, 1, 2], parent=[3, 3, 4, 4, -1],
               strength=[0, 0, 0, 0.5, 0.7], **common)
    with pytest.raises(ValueError, match="one label per runner"):
        race_probabilities(np.array([0.0, 1.0]), structure=bad)
