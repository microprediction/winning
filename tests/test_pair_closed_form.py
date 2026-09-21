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


