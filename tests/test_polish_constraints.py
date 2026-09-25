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
