"""Unused levels of a categorical group column contribute no rows (#933).

pandas lists every category in a groupby, an unused one with an empty
position array, and the calibration kernels refused the empty group: a
frame filtered down from a larger one crashed in every grouped adapter.
"""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from winning.classic.lattice import skew_normal_density
from winning.classic.pandas_util import (add_ability_implied_state_price_to_dataframe,
                                         add_centered_ability_to_dataframe,
                                         add_skew_normal_ability_to_dataframe)

D = skew_normal_density(L=15, unit=.2, scale=1, a=0)

CALLS = {
    "forward": lambda df: add_ability_implied_state_price_to_dataframe(df, "ability", "race", D, "q", .2),
    "inverse": lambda df: add_centered_ability_to_dataframe(df, "p", "race", D, .2, "mu"),
    "skew": lambda df: add_skew_normal_ability_to_dataframe(df, "race", prob_col="p", new_col="mu"),
}


def _frame(categories, index=None):
    return pd.DataFrame({
        "race": pd.Categorical(["A", "A", "B", "B", "A"], categories=categories),
        "ability": [0., 1., -1., 2., .5],
        "p": [.3, .5, .2, .8, .2],
    }, index=index)


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("categories", [["A", "B", "unused"], ["unused", "A", "B", "gone"]])
@pytest.mark.parametrize("index", [None, [4, 4, 9, 9, 4]])
def test_unused_categories_match_the_trimmed_frame(name, categories, index):
    df = _frame(categories, index)
    trimmed = df.copy()
    trimmed["race"] = trimmed["race"].cat.remove_unused_categories()
    got, want = CALLS[name](df.copy()), CALLS[name](trimmed)
    col = got.columns[-1]
    np.testing.assert_array_equal(got[col].to_numpy(), want[col].to_numpy())
    assert np.isfinite(got[col].to_numpy()).all()
    # the caller's column and index come back as they went in
    assert list(got["race"].cat.categories) == categories
    assert list(got.index) == list(df.index)
