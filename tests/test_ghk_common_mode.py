"""The dense route prices the race, not the coordinate system.

Adding `a 11'` to a covariance adds one shared Gaussian shift to every
performance. It cancels from every ordering, so the race is unchanged
and the choice-relevant quantity -- the contrast variance
`C_ii + C_jj - 2 C_ij` -- is unchanged with it.

The dense route factored the covariance, handed the factor to
`qmc_ghk`, and let it rebuild `L @ L.T`. That round trip is not
lossless: reconstructing from a matrix dominated by an unidentifiable
common mode loses exactly the small choice-relevant eigenvalue. The
winner moved by 0.033 while the stored contrast stayed exactly 2.0
(#302). The R port always formed the contrast from the original matrix.
"""
import numpy as np
import pytest

from scipy.special import ndtr

from winning.factor.races import race_probabilities


def _two_runner(a):
    return np.eye(2) + a * np.ones((2, 2))


@pytest.mark.parametrize("a", [0.0, 1e3, 3e14, 2e15, 4e15])
def test_a_common_mode_does_not_move_the_binary_race(a):
    """Closed form: with contrast variance 2 and a gap of 1, the winner
    is Phi(1/sqrt(2)) whatever the common mode."""
    C = _two_runner(a)
    assert C[0, 0] + C[1, 1] - 2 * C[0, 1] == pytest.approx(2.0)
    p = race_probabilities(np.array([0.0, 1.0]), cov=C)
    assert p[0] == pytest.approx(float(ndtr(1 / np.sqrt(2))), abs=1e-9)


def test_the_error_it_used_to_make_was_large():
    """Guard the guard: if the fixture stopped being a hard case this
    test would pass against the defect.

    The contrast has to be evaluated EXACTLY to see the loss. In float64
    `R00 + R11 - 2 R01` on entries of size 4e15 cancels to exactly 2.0
    and the reconstruction looks perfect; summed as rationals it is 1.5,
    a 25% error. The Cholesky's conditional-variance step is where it
    goes: `C22 - L21**2` is catastrophic cancellation at this scale.

    What `L22**2` comes out as is NOT asserted. It is the LAPACK
    build's rounding, not the defect: Accelerate gives 1.4030 and
    OpenBLAS (every Linux and Windows CI job) gives 1.5, against an
    exact 2.0 (#316). Both lose the contrast, which is what this
    guards.
    """
    from fractions import Fraction

    C = _two_runner(4e15)
    L = np.linalg.cholesky(C)
    # the conditional variance is materially wrong, whatever the build
    assert abs(L[1, 1] ** 2 - 2.0) > 0.1
    R = [[Fraction(x) for x in row] for row in (L @ L.T).tolist()]
    exact = float(R[0][0] + R[1][1] - 2 * R[0][1])
    assert abs(exact - 2.0) > 0.1, (
        "the round trip no longer loses the contrast, so this fixture "
        "no longer tests anything")
    # and the loss reaches the price: the old route, factor then
    # rebuild, misprices the binary race the fixed route gets exactly
    from winning.methods.native import qmc_ghk
    old, _ = qmc_ghk(np.array([0.0, -1.0]), L, np.zeros(2))
    assert abs(old[0] - float(ndtr(1 / np.sqrt(2)))) > 0.01
    # and float64 hides it, which is why this is measured in Fractions
    Rf = L @ L.T
    assert Rf[0, 0] + Rf[1, 1] - 2 * Rf[0, 1] == pytest.approx(2.0)


def test_beyond_float64_the_input_itself_has_no_contrast():
    """Not a defect to fix: at a = 1e16 the STORED matrix has contrast
    exactly zero, because 1 + 1e16 == 1e16. The race is undetermined by
    its own input, and no route can recover it. Stated here so the
    boundary is a measurement rather than a surprise."""
    C = _two_runner(1e16)
    assert C[0, 0] + C[1, 1] - 2 * C[0, 1] == 0.0


def test_a_dense_covariance_is_unmoved():
    """The fix must be a no-op wherever the old path was fine."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    C = A @ A.T + np.eye(8)
    mu = rng.normal(size=8)
    p = race_probabilities(mu, cov=C)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    shifted = race_probabilities(mu, cov=C + 1e6 * np.ones((8, 8)))
    assert np.abs(p - shifted).max() < 1e-9


def test_the_cov_door_does_not_weaken_the_loadings_door():
    """The registry skips shape normalisation only for an explicit
    cov=; every other call still goes through as_loadings/as_idio."""
    from winning.methods.native import qmc_ghk
    mu = np.zeros(3)
    with pytest.raises(ValueError):
        qmc_ghk(mu, np.zeros((3, 1)), np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        qmc_ghk(mu, np.zeros((3, 1)), np.ones(3), cov=np.eye(2))


_ASYM = np.array([[1.0, .8, .1],
                  [.2, 1.0, .4],
                  [.3, .6, 1.0]])


@pytest.mark.parametrize("C, match", [
    (_ASYM, "not symmetric"),
    (_ASYM.T, "not symmetric"),
    (np.full((3, 3), np.nan), "NaN or inf"),
    (np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
     "not positive semidefinite"),
    (np.ones((3, 2)), "square"),
    (np.eye(2), "3x3"),
])
def test_the_cov_door_validates_the_matrix(C, match):
    """The registry lets cov= skip as_loadings/as_idio because the method
    validates the matrix itself. It checked only the shape: an
    asymmetric C and its transpose priced two different races and a
    NaN matrix came back as NaN probabilities (#102)."""
    from winning.methods.native import qmc_ghk
    with pytest.raises(ValueError, match=match):
        qmc_ghk(np.zeros(3), None, None, cov=C)


def test_the_cov_door_refuses_loadings_alongside():
    """cov= replaces V and D. Handed both, it silently priced cov and
    dropped the rest (#102)."""
    from winning.methods.native import qmc_ghk
    mu = np.zeros(3)
    with pytest.raises(ValueError, match="pass one only"):
        qmc_ghk(mu, np.zeros((3, 1)), None, cov=np.eye(3))
    with pytest.raises(ValueError, match="pass one only"):
        qmc_ghk(mu, None, np.ones(3), cov=np.eye(3))
    p, _ = qmc_ghk(mu, None, None, cov=np.eye(3))
    assert p == pytest.approx(np.full(3, 1 / 3), abs=1e-3)
