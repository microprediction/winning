"""The (n, rank) loading contract at the Python boundary (issue #66).

A bare length-n vector of loadings used to be read by np.atleast_2d as
(1, n) -- ONE contestant carrying n factors. Nothing downstream noticed:
the gauge-fix subtracted the single row from itself, so the loadings
became zero and the rank was read as n. With the compiled kernel that
ended in an ndarray PANIC, which is not an Exception and so escaped the
try/except around a sweep; on the pure-numpy path it was worse, silently
pricing the INDEPENDENT race.

These tests pin both halves: equivalent spellings of the same loadings
agree exactly, and a shape that is not a loading matrix raises a
ValueError naming (n, rank) instead of reaching the compiled side.

The loadings below are deliberately NON-CONSTANT. A constant column is
gauge-fixed to zero, so a constant V cannot tell the factor race apart
from the independent one and cannot see this bug at all -- which is why
the original report's repro produced a panic rather than a wrong number.
"""
import numpy as np
import pytest

import winning.factor as wf
from winning.factor.core import as_idio, as_loadings
from winning.factor.permutations import ordered_probabilities
from winning.factor.structures import Blocks, Factor, Independent

N = 6
MU = np.linspace(-0.5, 0.5, N)
V1 = np.array([0.7, -0.3, 0.15, 0.55, -0.62, 0.02])   # rank one, non-constant
D = np.full(N, 0.96)


def _p(**kw):
    return np.asarray(wf.race_probabilities(MU, points=257, **kw))


def test_loadings_bite_so_the_comparison_is_not_vacuous():
    """Guard the guard: if V1 did not move the race, every equality
    below would hold for the wrong reason."""
    assert np.abs(_p(V=V1.reshape(N, 1), D=D) - _p(V=None, D=D)).max() > 1e-2


@pytest.mark.parametrize("spelling", ["1d", "col", "row"])
def test_every_spelling_of_the_same_loadings_prices_one_race(spelling):
    V = {"1d": V1, "col": V1.reshape(N, 1), "row": V1.reshape(1, N)}[spelling]
    assert np.abs(_p(V=V, D=D) - _p(V=V1.reshape(N, 1), D=D)).max() == 0.0


def test_rank_two_is_transposed_to_the_contract_shape():
    V2 = np.array([[0.9, -0.2], [0.1, 0.7], [-0.5, 0.3],
                   [0.4, -0.6], [0.0, 0.2], [-0.3, -0.1]])
    assert np.abs(_p(V=V2, D=D) - _p(V=V2.T, D=D)).max() == 0.0


def test_a_scalar_loading_is_a_common_column_and_cannot_move_the_race():
    """Not a special case: a loading every contestant shares adds the
    same c*f to every performance and cannot move an argmin."""
    assert np.abs(_p(V=0.35, D=D) - _p(V=None, D=D)).max() < 1e-12


def test_pure_python_path_agrees_with_the_compiled_one(monkeypatch):
    """The regression that had no symptom: without fastrace the 1-D V
    used to return the independent race to machine precision.

    _HAVE_RUST is read per call, so setattr is enough -- reloading the
    module to re-read WINNING_PURE rebinds every function in it and
    leaves other modules holding the old objects.
    """
    import winning.factor.races as races
    monkeypatch.setattr(races, "_HAVE_RUST", False)
    got = np.asarray(races.race_probabilities(MU, V=V1, D=D, points=257))
    want = np.asarray(races.race_probabilities(MU, V=V1.reshape(N, 1), D=D,
                                               points=257))
    ind = np.asarray(races.race_probabilities(MU, V=None, D=D, points=257))
    assert np.abs(got - want).max() == 0.0
    assert np.abs(got - ind).max() > 1e-2           # NOT the independent race


def test_the_inverse_and_the_ordered_prefixes_inherit_the_contract():
    p = _p(V=V1.reshape(N, 1), D=D)
    m1 = np.asarray(wf.abilities_from_race(p, V=V1, D=D, points=257))
    m2 = np.asarray(wf.abilities_from_race(p, V=V1.reshape(N, 1), D=D,
                                           points=257))
    assert np.abs(m1 - m2).max() == 0.0
    o1 = ordered_probabilities(MU, k=2, V=V1, D=D, points=301)
    o2 = ordered_probabilities(MU, k=2, V=V1.reshape(N, 1), D=D, points=301)
    assert np.abs(o1 - o2).max() == 0.0


def test_the_tempered_path_inherits_the_contract():
    a = _p(V=V1, D=D, temperature=0.4)
    b = _p(V=V1.reshape(N, 1), D=D, temperature=0.4)
    assert np.abs(a - b).max() == 0.0


@pytest.mark.parametrize("bad", [
    np.full(4, 0.2),                 # 1-D, wrong length
    np.full((2, N, 1), 0.2),         # 3-D
    np.full((4, 3), 0.2),            # 2-D, neither axis is the field
])
def test_a_mis_shaped_V_raises_before_the_compiled_side(bad):
    with pytest.raises(ValueError, match=r"one row per contestant"):
        wf.race_probabilities(MU, V=bad, D=D, points=129)


def test_a_mis_shaped_D_raises_before_the_compiled_side():
    with pytest.raises(ValueError, match=r"one idiosyncratic variance"):
        wf.race_probabilities(MU, V=V1, D=np.full(4, 0.96), points=129)


def test_normalisers_are_shape_only():
    assert as_loadings(V1, N).shape == (N, 1)
    assert np.array_equal(as_loadings(V1, N).ravel(), V1)
    assert as_loadings(np.zeros((N, 0)), N).shape == (N, 0)   # empty = indep
    assert np.array_equal(as_idio(0.5, 3), np.full(3, 0.5))


def test_ordered_probabilities_accepts_the_declarative_covariances():
    """Issue #66's second point: a covariance described once for the
    forward call is reusable here, and the grammars it cannot price say
    so by name rather than quietly answering a different question."""
    want = ordered_probabilities(MU, k=2, V=V1.reshape(N, 1), D=D, points=301)
    got = ordered_probabilities(MU, k=2, points=301,
                                structure=Factor(V=V1.reshape(N, 1), D=D))
    assert np.abs(got - want).max() == 0.0

    ind = ordered_probabilities(MU, k=2, D=D, points=301)
    assert np.abs(ordered_probabilities(MU, k=2, points=301,
                                        structure=Independent(D=D))
                  - ind).max() == 0.0

    # cov= must route through the same fit the forward call uses, so
    # k=1 here and race_probabilities(cov=) are one number. (That fit
    # can return a near-zero D and therefore a very fine lattice; the
    # warning is fit_covariance's, not this call's, and the mass check
    # below it is what decides accuracy.)
    Dv = np.linspace(0.8, 1.2, N)
    cov = np.outer(V1, V1) + np.diag(Dv)
    with pytest.warns(RuntimeWarning, match=r"too wide to resolve"):
        by_cov = ordered_probabilities(MU, k=1, cov=cov, points=301)
    fwd = np.asarray(wf.race_probabilities(MU, cov=cov, points=257))
    assert np.abs(by_cov - fwd).max() < 5e-3

    with pytest.raises(NotImplementedError, match=r"factor lattice"):
        ordered_probabilities(MU, k=2, points=301,
                              structure=Blocks(cluster=np.zeros(N, int),
                                               loading=np.full(N, 0.3), D=D))
    with pytest.raises(ValueError, match=r"pass one only"):
        ordered_probabilities(MU, k=2, V=V1, D=D, points=301,
                              structure=Independent(D=D))


def test_Factor_takes_V_and_D_by_keyword():
    """The names in the signature are the names that work (issue #66)."""
    assert Factor(V=V1.reshape(N, 1), D=D).n == N
    with pytest.raises(TypeError):
        Factor(loadings=V1.reshape(N, 1), idio=D)
