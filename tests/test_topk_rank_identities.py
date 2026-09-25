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


# a second field, from #221: sds spanning 232x. When #221 was written its
# RAW quadrature was off by 0.249 in a row, and dividing by those same row
# sums left a column defect of 0.0047 -- inside the 5e-3 tolerance -- so
# the post-normalisation check alone returned a false success on it.
#
# #224 then refined the lattice to resolve the NARROWEST runner, and this
# field, and the #217 field above, now resolve cleanly at every requested
# resolution: 4.5e-9 and 3.5e-9 respectively, where they used to be 4.7e-3
# and worse. That is the fix working, so the tests below record the new
# truth rather than pinning the old symptom.
MU_RAW = np.array([1.7802347516623231, 7.71387298475329,
                   -8.864593234372917, -13.723882426584884])
SD_RAW = np.array([0.09388844913109345, 0.2459800972322556,
                   1.0565710909345742, 21.775572080525823])

# A field the refinement CANNOT rescue: sds spanning 8636x, so the point
# count it asks for is past the 8193 cap and the raw quadrature really is
# defective. This is what still exercises the pre-normalisation guard.
MU_UNRESOLVABLE = np.array([-5.629882, -10.123372, -4.986196, 0.330608])
SD_UNRESOLVABLE = np.array([12.4970445, 0.00144703, 3.06481372, 0.00690731])


@pytest.mark.parametrize("points", [513, 1001])
def test_a_gross_raw_defect_is_caught_before_normalisation(points):
    """The guard #221 added, on a field past the refinement cap."""
    with pytest.raises(RuntimeError, match="before normalisation"):
        rank_probabilities(MU_UNRESOLVABLE, D=SD_UNRESOLVABLE ** 2,
                           points=points)


@pytest.mark.parametrize("mu,sd,name", [
    (MU_RAW, SD_RAW, "#221"),
    (MU, SD, "#217"),
])
def test_the_fields_that_used_to_be_defective_now_resolve(mu, sd, name):
    """#224's refinement is why. Both guards stay -- they are cheap and
    they are independent -- but neither fires on these any more, and
    pinning them as failures would pin a bug that is fixed."""
    P = rank_probabilities(mu, D=sd ** 2, points=513)
    assert np.abs(P.sum(axis=0) - 1).max() < 1e-6, name
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-12, name


def test_the_returned_matrix_guard_is_now_a_backstop():
    """Honest statement of where the two guards stand after #224.

    The RETURNED-matrix guard catches a raw matrix that passes its own
    check and then has its columns moved past tolerance by row
    normalisation. With the lattice refined to the narrowest runner, a
    search over 1200 deliberately extreme fields -- three to five runners,
    sds spanning 1e-4 to 80 -- trips the RAW guard 903 times and the
    RETURNED guard not once. It stays because it is one comparison and it
    is independent of the other, not because a field is known to reach it.
    """
    import warnings as _w
    rng = np.random.default_rng(7)
    raw = returned = 0
    for _ in range(200):
        n = int(rng.integers(3, 6))
        mu = rng.normal(scale=float(rng.uniform(1, 12)), size=n)
        sd = np.exp(rng.uniform(np.log(1e-4), np.log(80), size=n))
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                rank_probabilities(mu, D=sd ** 2, points=513)
        except RuntimeError as e:
            if "before normalisation" in str(e):
                raw += 1
            elif "RETURNED matrix" in str(e):
                returned += 1
    assert raw > 0, "the raw guard should still be reachable"
    assert returned == 0, (
        "a field now reaches the RETURNED guard -- update this test and "
        "say which, rather than leaving the claim stale")


def test_the_raw_field_resolves_and_agrees_with_top_k():
    P = rank_probabilities(MU_RAW, D=SD_RAW ** 2, points=2001)
    q = np.asarray(top_k_probabilities(MU_RAW, 2, D=SD_RAW ** 2, points=2001))
    assert np.abs(P[:, :2].sum(axis=1) - q).max() < 1e-4
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3


def test_rank_and_top_k_agree_about_whether_a_field_resolves():
    """The user-visible symptom of #221: rank returned while top-k rejected
    the same field at the same points. Checked on the field that still
    cannot resolve, since that is where the two can still disagree."""
    for points in (513, 1001):
        rank_failed = top_k_failed = False
        try:
            rank_probabilities(MU_UNRESOLVABLE, D=SD_UNRESOLVABLE ** 2,
                               points=points)
        except RuntimeError:
            rank_failed = True
        try:
            top_k_probabilities(MU_UNRESOLVABLE, 2,
                                D=SD_UNRESOLVABLE ** 2, points=points)
        except Exception:
            top_k_failed = True
        assert rank_failed == top_k_failed, (
            f"at points={points} rank and top-k disagree about resolution")


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
