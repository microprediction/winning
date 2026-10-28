"""The DEFAULT race rules measured against winning.certified.

Optional: needs python-flint (pip install winning[certified]); without
it the enclosure tests skip. Each tolerance is about ten times the error
measured when it was set (in the comments), so a regression in the
default lattice or the default factor node rule fails here, and an
improvement does not.
"""

import numpy as np
import pytest

from winning.factor.races import kink_lattice, race_probabilities

FIELDS = {
    "A4": ([-0.5, 0.0, 0.2, 0.3], [1.0, 0.6, 1.4, 1.0]),
    "B6": ([-0.802, -1.324, -0.248, 0.42, 1.136, 0.11],
           [1.038, 0.843, 0.869, 0.874, 1.487, 1.133]),
    "C3": ([0.0, 0.0, 0.0], [0.5, 1.0, 2.0]),
}

# independent race, default 257-point lattice. Measured max |error| over
# A4, B6, C3: normal 5.6e-17, gumbel 6.2e-14, logistic 1.9e-14, laplace
# 4.9e-4 (the uniform lattice is second order across the laplace kink);
# laplace with kink_split=True 5.0e-14
INDEPENDENT_TOL = {"normal": 1e-14, "gumbel": 1e-12, "logistic": 1e-12,
                   "laplace": 5e-3}
KINK_SPLIT_TOL = 5e-13

# factor race, default node rule, digits=12 certificates. Measured:
#   F5 (five runners, rank one): normal 4.9e-7, gumbel 3.0e-5,
#       logistic 2.2e-5 (the 15-node Gauss-Hermite rule)
#   F4 (four runners, rank one): normal 3.5e-13, gumbel 2.4e-9,
#       logistic 6.9e-10
#   F3r2 (three runners, rank two, digits=9): normal 3.6e-10
_rng = np.random.default_rng(11)
FACTOR_FIELDS = {
    "F5": (_rng.normal(0, 1, 5), _rng.normal(0, 0.6, (5, 1)),
           _rng.uniform(0.5, 1.5, 5)),
    "F4": ([-0.3, 0.0, 0.1, 0.4], [[0.5], [-0.3], [0.2], [0.6]],
           [0.8, 1.0, 1.2, 1.0]),
    "F3r2": ([0.0, 0.2, -0.1], [[0.6, 0.1], [-0.4, 0.5], [0.2, -0.5]],
             [0.5, 1.0, 1.5]),
}
FACTOR_TOL = {
    ("F5", "normal"): 5e-6, ("F5", "gumbel"): 3e-4, ("F5", "logistic"): 2e-4,
    ("F4", "normal"): 5e-12, ("F4", "gumbel"): 3e-8, ("F4", "logistic"): 1e-8,
    ("F3r2", "normal"): 5e-9,
}


def _mid(balls):
    return np.array([float(b.mid()) for b in balls])


@pytest.mark.parametrize("field", sorted(FIELDS))
@pytest.mark.parametrize("base", sorted(INDEPENDENT_TOL))
def test_default_independent_rule_within_bound(field, base):
    pytest.importorskip("flint")
    from winning.certified import certified_race_probabilities
    mu, D = FIELDS[field]
    sure = _mid(certified_race_probabilities(mu, D, base=base, digits=20))
    err = np.abs(race_probabilities(mu, D=D, base=base) - sure).max()
    assert err < INDEPENDENT_TOL[base], (field, base, err)


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_kink_split_laplace_within_bound(field):
    pytest.importorskip("flint")
    from winning.certified import certified_race_probabilities
    mu, D = FIELDS[field]
    sure = _mid(certified_race_probabilities(mu, D, base="laplace",
                                             digits=20))
    err = np.abs(race_probabilities(mu, D=D, base="laplace",
                                    kink_split=True) - sure).max()
    assert err < KINK_SPLIT_TOL, (field, err)


@pytest.mark.parametrize("field, base", sorted(FACTOR_TOL))
def test_default_factor_node_rule_within_bound(field, base):
    pytest.importorskip("flint")
    from winning.certified import certified_factor_race_probabilities
    mu, V, D = FACTOR_FIELDS[field]
    digits = 9 if field == "F3r2" else 12
    sure = _mid(certified_factor_race_probabilities(mu, V, D, base=base,
                                                    digits=digits))
    err = np.abs(race_probabilities(mu, V=np.asarray(V, dtype=float), D=D,
                                    base=base) - sure).max()
    assert err < FACTOR_TOL[(field, base)], (field, base, err)


# --- the opt-in path's cost bound and contract (no flint needed) ---------

def test_kink_lattice_node_count_is_bounded():
    # a thousand kinks inside a 257-point window: merged onto len(x) // 4
    # bins, so the split lattice stays near the uniform budget
    x = np.linspace(-10.0, 10.0, 257)
    M = np.random.default_rng(0).normal(0, 2, 1000)
    nodes, weights = kink_lattice(x, M, np.ones(1000), (0.0,))
    assert len(nodes) < 2 * len(x)
    assert abs(weights.sum() - 20.0) < 1e-12


def test_kink_lattice_small_field_keeps_every_kink():
    x = np.linspace(-10.0, 10.0, 257)
    M = np.array([-0.5, 0.0, 0.2, 0.3])
    nodes, _ = kink_lattice(x, M, np.ones(4), (0.0,))
    # Gauss-Legendre nodes are interior, so no node sits on a kink, and
    # every piece is bounded by two kinks or the window ends
    for k in M:
        assert not np.any(nodes == k)
        assert np.any(nodes < k) and np.any(nodes > k)


def test_kink_split_default_is_off_and_smooth_bases_ignore_it():
    mu, D = FIELDS["A4"]
    for base in ("normal", "gumbel", "logistic"):
        a = race_probabilities(mu, D=D, base=base)
        b = race_probabilities(mu, D=D, base=base, kink_split=True)
        assert np.array_equal(a, b)
    a = race_probabilities(mu, D=D, base="laplace")
    b = race_probabilities(mu, D=D, base="laplace", kink_split=True)
    assert np.abs(a - b).max() > 1e-6     # it does something when asked


def test_kink_split_refuses_structure():
    from winning.factor.structures import Independent
    with pytest.raises(ValueError, match="kink_split"):
        race_probabilities([0.0, 0.1, 0.2], structure=Independent(D=[1.0, 1.0, 1.0]),
                           base="laplace", kink_split=True)
