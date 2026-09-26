"""`structure=` describes the covariance; it does not silence the knobs.

`race_probabilities(..., structure=...)` dropped `points`, `window` and
`delta` at the front door, and `dispatch_probabilities` dropped `points`
a second time recursing into `Independent` and `Factor` -- it is a NAMED
parameter there, so it never reached `**kw`. A caller asking for more
resolution got the default silently: `points=17`, `257` and `2049`
returned byte-identical answers on a Blocks race (#89).

The same front door accepted `structure=` together with `V=/D=/F=/W=`
and discarded some of them, with forward and inverse discarding
different ones.
"""
import numpy as np
import pytest

from winning.factor.races import race_probabilities
from winning.factor.structures import Blocks, Factor, Independent

MU = np.array([0.0, 0.3, -0.2, 0.5])
D4 = np.ones(4)
BLOCKS = Blocks(cluster=np.array([0, 0, 1, 1]),
                loading=np.array([0.5, 0.4, 0.3, 0.2]), D=D4)
FACTOR = Factor(np.array([[0.5], [0.4], [-0.3], [0.2]]), D4)
INDEP = Independent(D4)


@pytest.mark.parametrize("structure", [BLOCKS, FACTOR],
                         ids=["Blocks", "Factor"])
def test_points_reaches_the_kernel(structure):
    """A coarse lattice must give a DIFFERENT answer from a fine one.
    Equality here is the bug, not the goal."""
    coarse = race_probabilities(MU, structure=structure, points=17)
    fine = race_probabilities(MU, structure=structure, points=2049)
    assert not np.allclose(coarse, fine, atol=1e-9), (
        "points= did not reach the kernel")
    mid = race_probabilities(MU, structure=structure, points=257)
    # and the fine answer is the converged one: 257 is nearer 2049 than
    # 17 is, or the knob is reaching something other than the lattice
    assert (np.abs(mid - fine).max() < np.abs(coarse - fine).max())


# A HETEROGENEOUS field, because the easy one is already converged: at
# D = ones(4) the answer moves by 1e-16 between points=65 and 257, so a
# comparison there passes whether or not points is forwarded at all. With
# one narrow runner it moves by 2.5e-3.
D_HET = np.array([1.0, 0.05, 1.0, 1.0])
INDEP_HET = Independent(D_HET)


def test_independent_structure_matches_the_plain_race():
    """The Independent branch drops through to the same race, so the
    forwarded controls must land in the same place -- checked at a
    resolution where they visibly matter."""
    coarse = race_probabilities(MU, structure=INDEP_HET, points=17)
    fine = race_probabilities(MU, structure=INDEP_HET, points=257)
    assert not np.allclose(coarse, fine, atol=1e-6), (
        "fixture is converged at 17 points; it cannot detect a dropped "
        "points= and must be made harder")
    for pts in (17, 65, 257):
        got = race_probabilities(MU, structure=INDEP_HET, points=pts)
        want = race_probabilities(MU, D=D_HET, points=pts)
        assert np.allclose(got, want, atol=1e-12), f"points={pts}"


def test_delta_reaches_the_kernel():
    """delta= moves the answer, so forwarding is demonstrated rather
    than assumed. It has to be the INDEPENDENT structure: on the factor
    structure delta moves this field by 3e-16, indistinguishable from a
    dropped argument, so that would have been an untestable claim."""
    base = race_probabilities(MU, structure=INDEP_HET)
    moved = race_probabilities(MU, structure=INDEP_HET, delta=1e-4)
    assert np.abs(moved - base).max() > 1e-11


@pytest.mark.parametrize("kw", [{"window": "span"}, {"delta": 1e-9}])
def test_the_factor_path_accepts_the_lattice_controls(kw):
    """Named for what it actually checks. `window="span"` is ACCEPTED
    and prices; it does not measurably move any fixture I could build
    (1e-16 on four of them), so this cannot demonstrate that it reached
    the kernel -- only that it is not refused, which is what separates
    the Factor branch from the hierarchical ones above."""
    p = race_probabilities(MU, structure=FACTOR, **kw)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("kw", [{"window": "span"}, {"delta": 1e-9}])
def test_hierarchical_kernels_refuse_what_they_cannot_honour(kw):
    """These choose their own window per cluster and take neither
    argument, so forwarding would drop them silently -- the same defect
    one level down. They are refused, as base/temperature already are."""
    with pytest.raises(NotImplementedError) as e:
        race_probabilities(MU, structure=BLOCKS, **kw)
    assert "lattice window" in str(e.value)


def test_the_refusal_explains_the_right_reason():
    """A model-level refusal and a lattice-level one have different
    reasons; one message for both would misexplain whichever it is."""
    with pytest.raises(NotImplementedError) as e:
        race_probabilities(MU, structure=BLOCKS, temperature=0.5)
    assert "probabilities only" in str(e.value)
    with pytest.raises(NotImplementedError) as e:
        race_probabilities(MU, structure=BLOCKS, window="span")
    assert "own lattice window" in str(e.value)


@pytest.mark.parametrize("name", ["V", "D", "F", "W"])
def test_structure_plus_a_second_description_is_refused(name):
    arg = {"V": np.ones((4, 1)), "D": D4,
           "F": np.zeros((1, 1)), "W": np.ones(1)}[name]
    with pytest.raises(ValueError, match="describes the covariance"):
        race_probabilities(MU, structure=FACTOR, **{name: arg})


def test_structure_alone_still_prices():
    for st in (INDEP, FACTOR, BLOCKS):
        p = race_probabilities(MU, structure=st)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
