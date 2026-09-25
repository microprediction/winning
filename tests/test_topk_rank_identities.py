"""The rank matrix that is RETURNED satisfies the identities it claims.

`rank_probabilities` checked the raw quadrature matrix, then divided by the
raw row sums and returned that. Row normalisation is not neutral: it makes
every row exact and MOVES the columns, so a matrix that passed the check
could fail the stated identity afterwards and nothing looked. On a field
whose sds span 73x it pushed the column excess from 4.5e-3 to 5.4e-3, and
the documented identity -- cumulative rank rows reproduce
top_k_probabilities -- broke by 2.6e-3 (#203).

The columns are deliberately NOT forced to sum to one. Alternating
row/column scaling would make both identities exact, and would do it by
hiding the under-resolution that caused the defect: the same field at 2001
points has a column error of 3.5e-9, so the quadrature is what is wrong,
not the normalisation.
"""
import pathlib
import warnings

import numpy as np
import pytest

from winning.factor.topk import rank_probabilities, top_k_probabilities

# the reported field: sds from 0.19 to 14.3, a 73x spread
MU = np.array([0.852479112355, 1.342705472309, -4.927823648082])
SD = np.array([0.635892136598, 0.194587863958, 14.30720080209])


def test_a_heterogeneous_field_resolves_itself_at_the_default():
    """#224: the grid is now refined to the NARROWEST runner, so this
    field no longer has to be rejected -- it is resolved. The guards
    below remain for what refinement cannot reach."""
    P = rank_probabilities(MU, D=SD ** 2, points=513)
    ref = rank_probabilities(MU, D=SD ** 2, points=8193)
    assert np.abs(P - ref).max() < 1e-8
    assert np.abs(P.sum(axis=0) - 1).max() < 1e-6


@pytest.mark.parametrize("points", [2001, 8001])
def test_both_identities_hold_on_the_returned_matrix(points):
    P = rank_probabilities(MU, D=SD ** 2, points=points)
    assert np.abs(P.sum(axis=1) - 1).max() < 5e-3      # a runner takes one rank
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3      # a rank is taken once
    assert np.isfinite(P).all()
    assert (P >= 0).all() and (P <= 1).all()


@pytest.mark.parametrize("k", [1, 2])
def test_cumulative_rows_reproduce_top_k(k):
    """The documented identity, which row normalisation had broken by
    2.6e-3 on this field: it now holds to the quadrature's own error."""
    P = rank_probabilities(MU, D=SD ** 2, points=2001)
    q = np.asarray(top_k_probabilities(MU, k, D=SD ** 2, points=2001))
    assert np.abs(P[:, :k].sum(axis=1) - q).max() < 1e-6


def test_an_ordinary_field_is_untouched():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=6)
    D = 0.5 + rng.random(6)
    P = rank_probabilities(mu, D=D, points=513)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-9
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3
    q = np.asarray(top_k_probabilities(mu, 2, D=D, points=513))
    assert np.abs(P[:, :2].sum(axis=1) - q).max() < 1e-6


# --- #224: the mass check is one scalar, and cannot see a runner ---------

def test_top_k_resolves_the_narrowest_runner():
    """The window is set by the WIDEST runner and the grid by `points`, so
    a heterogeneous field left the narrowest density between samples: sds
    of 0.093 and 6.02 at 513 points gave a spacing of 0.174, nearly twice
    the narrow runner's whole sd, and its membership was 4.4e-3 wrong."""
    mu = np.array([0.25185093, -1.43227624, -0.38996506,
                   -0.14749401, -0.95856037])
    sd = np.array([0.09336325, 2.29011235, 2.30865821,
                   1.80430574, 6.02449279])
    # independent one-dimensional Gaussian integration of
    # 1 - int f_i(x) prod_{j!=i} F_j(x) dx
    ref = np.array([0.84005969, 0.90113651, 0.79813532,
                    0.80744990, 0.65321859])
    q = np.asarray(top_k_probabilities(mu, 4, D=sd ** 2, points=513))
    assert np.abs(q - ref).max() < 1e-6


def test_top_k_is_resolution_independent_once_refined():
    """The signature of a resolved integral: asking for more points does
    not move it. Before, 513 and 2049 differed by 4.4e-3."""
    mu = np.array([0.25185093, -1.43227624, -0.38996506,
                   -0.14749401, -0.95856037])
    sd = np.array([0.09336325, 2.29011235, 2.30865821,
                   1.80430574, 6.02449279])
    a = np.asarray(top_k_probabilities(mu, 4, D=sd ** 2, points=513))
    b = np.asarray(top_k_probabilities(mu, 4, D=sd ** 2, points=2049))
    assert np.abs(a - b).max() < 1e-9


def test_the_mass_check_alone_would_not_have_caught_it():
    """Why refinement, and not a tighter tolerance: the memberships summed
    to 3.99440 against a tolerance of 0.02, comfortably inside, because
    runner-level errors of opposite sign cancel in one scalar."""
    mu = np.array([0.25185093, -1.43227624, -0.38996506,
                   -0.14749401, -0.95856037])
    sd = np.array([0.09336325, 2.29011235, 2.30865821,
                   1.80430574, 6.02449279])
    q = np.asarray(top_k_probabilities(mu, 4, D=sd ** 2, points=513))
    assert abs(q.sum() - 4) < 0.02        # the check that passed before
    ref = np.array([0.84005969, 0.90113651, 0.79813532,
                    0.80744990, 0.65321859])
    assert np.abs(q - ref).max() < 1e-6   # and the per-runner truth now


@pytest.mark.parametrize("smin", [1e-10, 1e-300])
def test_the_cap_is_reached_without_overflowing_on_the_way(smin):
    """#228: R and Julia computed the uncapped requirement in a
    fixed-width integer and overflowed BEFORE the `min(need, 8193)` --
    in the very regime the cap exists for. R's as.integer() gave NA, so
    the `if (need > 8193)` meant to warn errored instead; Julia threw
    InexactError. Python is unaffected only because its integers are
    unbounded, which is exactly why the ports needed their own check.
    """
    from winning.factor.topk import _resolved_points
    got = _resolved_points(-18.0689583389, 11.4380329712,
                           np.array([smin, 1.0, 2.0]), 513)
    assert got == 8193


def test_the_ports_clamp_in_floating_point_before_converting():
    """Pinned in the source, because this machine cannot run every port
    and the failure is a conversion, not a value."""
    root = pathlib.Path(__file__).resolve().parents[1]
    r = (root / "r" / "winning" / "R" / "topk.R").read_text()
    assert "as.integer(min(need, 8193))" in r, \
        "R must clamp in double and convert after"
    assert "!is.finite(need)" in r
    jl = (root / "julia" / "winning" / "src" / "topk.jl").read_text()
    assert "Int(min(need, 8193.0))" in jl, \
        "julia must clamp in Float64 and convert after"
    assert "!isfinite(need)" in jl
