"""Group indices name members, not offsets from the end.

`concentration_matrix` wrote group membership with
`r[np.asarray(idx, int)] = 1.0`, so numpy read a NEGATIVE index as
counting from the end and silently capped the LAST name instead of the
one the caller meant. The browser, where a negative index is stored as a
non-element array property, silently dropped the member instead. Neither
is documented, and the two ports therefore meant different things by the
same input, so both refuse it now (#247).

An out-of-range POSITIVE index already raised here, and in the browser it
wrote past the row and returned NaN for every runner while reporting the
result feasible.

A non-finite `name_caps` ENTRY is a different matter: "NaN/None entries
skipped" is the documented contract, meaning no cap for that name, so it
stays a feature in both ports.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor.polish import concentration_matrix, polish_race


@pytest.mark.parametrize("bad", [-1, -3, 3, 99])
def test_a_group_index_outside_the_field_is_refused(bad):
    with pytest.raises(IndexError, match=r"not in \[0, 3\)"):
        concentration_matrix(3, groups=[([0, bad], 0.5)])


def test_the_message_names_the_offending_index():
    with pytest.raises(IndexError, match="group index -1"):
        concentration_matrix(3, groups=[([0, -1], 0.5)])


def test_a_valid_group_still_builds_one_row():
    A, b = concentration_matrix(3, groups=[([0, 1], 0.45)])
    assert A.shape == (1, 3)
    assert np.array_equal(A[0], [1.0, 1.0, 0.0])
    assert b[0] == 0.45


def test_a_nan_cap_entry_still_means_no_cap_for_that_name():
    A, b = concentration_matrix(3, name_caps=[0.3, np.nan, 0.4])
    assert A.shape == (2, 3)                 # the middle name is skipped
    assert np.array_equal(A[0], [1.0, 0.0, 0.0])
    assert np.array_equal(A[1], [0.0, 0.0, 1.0])


def test_a_wrong_length_name_caps_is_still_refused_by_broadcasting():
    for bad in ([0.3, 0.3], [0.3] * 4):
        with pytest.raises(ValueError):
            concentration_matrix(3, name_caps=bad)


def test_a_group_cap_actually_binds():
    """Without this the refusals above could hold on a builder that
    produces rows nothing reads."""
    res = polish_race(p0=np.array([0.2, 0.3, 0.5]), D=np.ones(3),
                      points=129, groups=[([0, 1], 0.45)])
    p = np.asarray(res.p if hasattr(res, "p") else res[0])
    assert abs(p[0] + p[1] - 0.45) < 1e-6


# ---------------------------------------------------------------- #671

@pytest.mark.parametrize("bad", [[1.9], [True], [np.nan], [np.inf], [0.5, 2],
                                 ["1"], [[0, 1]]])
def test_group_members_must_be_whole_numbers_before_any_cast_671(bad):
    # np.asarray(idx, int) truncated 1.9 to 1 and read True as 1, so a typo
    # constrained a different name and polish_race certified it feasible
    with pytest.raises(ValueError, match="group members"):
        concentration_matrix(3, groups=[(bad, 0.2)])


@pytest.mark.parametrize("bad", [[-1], [3]])
def test_group_members_out_of_range_still_refused_671(bad):
    with pytest.raises(IndexError):
        concentration_matrix(3, groups=[(bad, 0.2)])


def test_polish_race_refuses_a_fractional_group_member_671():
    with pytest.raises(ValueError, match="group members"):
        polish_race(p0=[0.1, 0.8, 0.1], D=np.ones(3), groups=[([1.9], 0.2)],
                    points=257)


def test_whole_number_members_in_any_numeric_dtype_are_the_same_group_671():
    want = concentration_matrix(4, groups=[([0, 2], 0.3)])
    for idx in ([0.0, 2.0], np.array([0, 2], dtype=np.int32), (0, 2), [],):
        A, b = concentration_matrix(4, groups=[(idx, 0.3)])
        if len(idx) == 0:
            assert A.shape == (1, 4) and not A.any()
            continue
        np.testing.assert_array_equal(A, want[0])
        np.testing.assert_array_equal(b, want[1])
