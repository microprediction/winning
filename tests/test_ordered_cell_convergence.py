"""A capped ordered-prefix lattice is policed per CELL, not by the total.

#269. `ordered_probabilities` sized its lattice by the widest runner and
sampled uniformly, so a field with sds of 22.0, 0.029 and 0.011 needed
259,498 points and got the 16,385-point cap. It then trusted one scalar
invariant: the total mass. That total was 1.00038 -- inside the default
`mass_tol=1e-3` -- while the (1, 2) cell was 2.1% high, because cell
errors of opposite sign cancel in a sum. Normalising spread the residual
and returned a plausible number.

`points=` was no help either: `min(max(points, need), cap)` is the cap
whenever `need` exceeds it, so every value of `points` gave the same
answer.

This is the ordered-prefix member of the family in #224, #244, #197 and
#228: a lattice spanned by the widest scale, sampled uniformly, with an
AGGREGATE guard that cannot see a per-element failure.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

import winning.factor.permutations as perm
from winning.factor.permutations import ordered_probabilities

# the reported fixture
MU = np.array([0.89065744, 0.64835238, 0.62197428])
SD = np.array([22.0223994, 0.0292273349, 0.0108711059])


def _exact_cell(i, j):
    """P(i first, j second) for independent normals, min-wins, by direct
    one-dimensional integration in runner j's own coordinate. The
    integrand is smooth apart from two Gaussian transitions, so the
    breakpoint is handed to quad rather than hoped for."""
    rest = [m for m in range(len(MU)) if m not in (i, j)]

    def integrand(z):
        x = MU[j] + SD[j] * z
        out = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        out = out * ndtr((x - MU[i]) / SD[i])          # i already finished
        for m in rest:
            out = out * ndtr((MU[m] - x) / SD[m])      # m still running
        return out

    brk = (MU[i] - MU[j]) / SD[j]
    val, err = quad(integrand, -14, 14, epsabs=1e-13, limit=400,
                    points=[brk])
    # quad's error estimate is conservative here (~1e-10 for a value
    # good to ~1e-15, as the cross-check below shows)
    assert err < 1e-9, err
    return val


def test_the_reference_integral_is_the_one_in_the_report():
    """Pin the oracle itself against the number derived independently
    in #269, so a later change to the integration cannot quietly move
    what everything below is measured against."""
    assert abs(_exact_cell(1, 2) - 0.10035114826251017) < 1e-13


def test_the_reported_cell_matches_the_exact_integral():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = ordered_probabilities(MU, k=2, D=SD ** 2)
    exact = _exact_cell(1, 2)
    rel = abs(q[1, 2] - exact) / exact
    assert rel < 1e-9, f"cell (1,2) = {q[1, 2]!r}, exact {exact!r}, rel {rel:.2e}"


def test_every_cell_matches_the_exact_integral():
    """Not just the reported one: the fix is per cell, so check them
    all. Anything else would pin the symptom."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = ordered_probabilities(MU, k=2, D=SD ** 2)
    worst = 0.0
    for i in range(3):
        for j in range(3):
            if i == j:
                assert q[i, j] == 0.0
                continue
            exact = _exact_cell(i, j)
            if exact > 1e-9:
                worst = max(worst, abs(q[i, j] - exact) / exact)
    assert worst < 1e-8, f"worst relative cell error {worst:.3e}"


def test_the_total_mass_could_not_have_caught_it():
    """The point of the whole change. At the capped lattice the total is
    inside the default tolerance while a cell is 2% wrong. Computed here
    independently of the package, so it pins the ARITHMETIC rather than
    our implementation of it."""
    pad = 8.0 * SD.max()
    x = np.linspace(MU.min() - pad, MU.max() + pad, 16385)
    dx = x[1] - x[0]
    z = (x[None, :] - MU[:, None]) / SD[:, None]
    S = 1.0 - ndtr(z)
    f = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) / SD[:, None]
    logS = np.log(np.maximum(S, 1e-300))
    Fc = 1.0 - S
    field = logS.sum(0)
    out = np.zeros((3, 3))
    for i in range(3):
        rest_i = np.exp(np.clip(field[None, :] - logS[i] - logS, -745.0, 0.0))
        contrib = (f * Fc[i][None, :] * rest_i).sum(1) * dx
        contrib[i] = 0.0
        out[i] = contrib

    assert abs(out.sum() - 1.0) < 1e-3          # the guard that passed
    exact = _exact_cell(1, 2)
    assert abs(out[1, 2] - exact) / exact > 0.02   # the cell that did not


def test_points_could_not_fix_it_either():
    """`min(max(points, need), cap)` is the cap once need exceeds it, so
    every `points=` gave the identical capped answer. After the fix the
    answer is the resolved one at any `points=`."""
    exact = _exact_cell(1, 2)
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in (301, 1025, 4097):
            seen.append(ordered_probabilities(MU, k=2, D=SD ** 2,
                                              points=p)[1, 2])
    for v in seen:
        assert abs(v - exact) / exact < 1e-9


def test_an_unresolvable_field_raises_rather_than_guessing(monkeypatch):
    """When the budget runs out before the cells settle, that is a
    deliberate error. Squeezing the budget so the fixture cannot reach
    the 32,769 points it needs is the cleanest way to reach that branch
    -- and it is exactly the state the old code returned from."""
    monkeypatch.setattr(perm, "_ORDERED_POINT_BUDGET", 3 * 20000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FloatingPointError) as e:
            ordered_probabilities(MU, k=2, D=SD ** 2)
    msg = str(e.value)
    assert "259498" in msg                       # what it needed
    assert "total mass is NOT evidence" in msg   # and why mass did not say so
    assert "cancel" in msg


def test_a_wide_field_says_so():
    with pytest.warns(RuntimeWarning, match="too wide to resolve"):
        ordered_probabilities(MU, k=2, D=SD ** 2)


def test_an_ordinary_field_neither_warns_nor_refines(recwarn):
    """The refinement must cost nothing when the lattice already
    resolves the field -- which is every ordinary call."""
    mu = np.array([0.0, 0.5, 1.0])
    q = ordered_probabilities(mu, k=2, D=np.ones(3), points=301)
    assert abs(q.sum() - 1.0) < 1e-12
    assert not [w for w in recwarn
                if "too wide to resolve" in str(w.message)]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_reported_field_is_a_distribution_at_every_k(k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = ordered_probabilities(MU, k=k, D=SD ** 2)
    assert abs(q.sum() - 1.0) < 1e-9
    assert (q >= 0).all()
