"""The pandas adapters calibrate by row position and keep the caller's index (#908).

A pandas index may repeat labels. The adapters wrote each group's answers
back with vals.loc[sub.index], which expands a duplicated label to every
row carrying it, so a frame with a legal non-unique index raised on the
length mismatch where the same rows with a reset index passed.
"""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from winning.classic.lattice import skew_normal_density
from winning.classic.lattice_conventions import STD_A, STD_L, STD_SCALE, STD_UNIT
from winning.classic.pandas_util import (add_ability_implied_state_price_to_dataframe,
                                         add_centered_ability_to_dataframe,
                                         add_skew_normal_ability_to_dataframe)

DENSITY = skew_normal_density(L=STD_L, unit=STD_UNIT, scale=STD_SCALE, a=STD_A)


def _frame(index):
    return pd.DataFrame({"race": ["A", "A", "B", "B", "A"],
                         "ability": [0.0, 1.0, -1.0, 2.0, 0.5],
                         "p": [0.3, 0.5, 0.2, 0.8, 0.2]}, index=index)


@pytest.mark.parametrize("index", [[0, 0, 1, 1, 0],            # repeats within and across groups
                                   ["x", "y", "x", "y", "x"],  # labels shared across groups
                                   [7, 7, 7, 7, 7]])           # one label for every row
def test_duplicate_labels_calibrate_row_by_row_and_keep_the_index(index):
    df = _frame(index)
    ref = _frame(None)                                 # the same rows under a RangeIndex
    q = add_ability_implied_state_price_to_dataframe(df, "ability", "race", DENSITY, "q", STD_UNIT)
    q_ref = add_ability_implied_state_price_to_dataframe(ref, "ability", "race", DENSITY, "q", STD_UNIT)
    mu = add_centered_ability_to_dataframe(df, "p", "race", DENSITY, STD_UNIT, "mu")
    mu_ref = add_centered_ability_to_dataframe(ref, "p", "race", DENSITY, STD_UNIT, "mu")
    sk = add_skew_normal_ability_to_dataframe(df, by="race", prob_col="p", new_col="mu")
    for got, want, col in ((q, q_ref, "q"), (mu, mu_ref, "mu"), (sk, mu_ref, "mu")):
        assert list(got.index) == list(index)                      # the caller's index, untouched
        np.testing.assert_allclose(got[col].to_numpy(), want[col].to_numpy(), atol=1e-12)
        assert np.isfinite(got[col].to_numpy()).all()
    # the issue's positional expectations for its two-by-two frame
    issue = pd.DataFrame({"race": ["A", "A", "B", "B"], "ability": [0.0, 1.0, -1.0, 2.0],
                          "p": [0.4, 0.6, 0.2, 0.8]}, index=[0, 0, 1, 1])
    q2 = add_ability_implied_state_price_to_dataframe(issue, "ability", "race", DENSITY, "q", STD_UNIT)
    np.testing.assert_allclose(q2["q"].to_numpy(), [0.760158, 0.239842, 0.983015, 0.016985], atol=1e-6)
    mu2 = add_centered_ability_to_dataframe(issue, "p", "race", DENSITY, STD_UNIT, "mu")
    # the inverse runs to CALIBRATION_TOL since #904; three sweeps gave 0.596095
    np.testing.assert_allclose(mu2["mu"].to_numpy(), [0.179342, -0.179342, 0.596096, -0.596096], atol=1e-6)


def test_the_answers_move_with_the_rows_not_the_labels():
    # two rows share a label but sit in different groups with different
    # inputs: each gets its own group's answer
    df = _frame([0, 1, 0, 1, 2])
    q = add_ability_implied_state_price_to_dataframe(df, "ability", "race", DENSITY, "q", STD_UNIT)["q"].to_numpy()
    assert abs(q[0] - q[2]) > 0.1 and abs(q[1] - q[3]) > 0.1
    # and groups sum to one
    assert abs(q[[0, 1, 4]].sum() - 1) < 1e-12 and abs(q[[2, 3]].sum() - 1) < 1e-12


def test_missing_group_label_stays_nan_and_the_input_is_not_mutated():
    df = _frame([0, 0, 1, 1, 2])
    df.loc[df.index == 2, "race"] = np.nan
    before = df.copy()
    out = add_centered_ability_to_dataframe(df, "p", "race", DENSITY, STD_UNIT, "mu")
    pd.testing.assert_frame_equal(df, before)
    assert np.isnan(out["mu"].to_numpy()[4]) and np.isfinite(out["mu"].to_numpy()[:4]).all()
