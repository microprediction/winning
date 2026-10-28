"""Dense-filter front-door contracts: one-shot iterables (#137) and the
walk-forward predictive using the filter's own base (#109)."""

import numpy as np
import pytest

from winning.ratings.history import (predict_race, rate_history,
                                     tune_history, walk_forward)

RACES = [
    {"t": 0.0, "runners": ["a", "b", "c", "d"], "order": [0, 1, 2, 3]},
    {"t": 1.0, "runners": ["a", "b", "c", "d"], "order": [2, 0, 1, 3]},
    {"t": 2.0, "runners": ["a", "c", "d"], "winner": 1},
]
# winner-only is Gaussian-only on the dense filter, so the base tests
# feed orders
ORDERS = RACES[:2] + [{"t": 2.0, "runners": ["a", "c", "d"],
                       "order": [1, 0, 2]}]


def test_a_generator_gives_the_same_answer_as_a_list():
    want = rate_history(RACES)
    got = rate_history(r for r in RACES)
    assert got[1] == want[1] != 0.0
    assert got[0] == want[0]


def test_tune_history_accepts_a_generator():
    want = tune_history(RACES, tune=("beta2",), maxiter=4)
    got = tune_history((r for r in RACES), tune=("beta2",), maxiter=4)
    assert got == want


@pytest.mark.parametrize("base", ["normal", "laplace"])
def test_walk_forward_scores_the_predictive_it_updates_under(base):
    ids = sorted({r for rc in ORDERS for r in rc["runners"]})
    out = walk_forward(ORDERS, warmup=1, base=base)
    state = None
    for i, race in enumerate(ORDERS):
        if i >= 1:
            p, _ = predict_race(state, race["runners"], t=race["t"],
                                base=base)
            rec = [r for r in out["records"] if r["i"] == i][0]
            assert np.allclose(rec["p_model"], p, rtol=0, atol=1e-15), \
                (base, i, rec["p_model"], p)
        kw = {"state": state} if state is not None else {"ids": ids}
        _, _, state = rate_history([race], return_state=True, base=base,
                                   **kw)


def test_the_base_moves_the_walk_forward_prediction():
    """Otherwise the equality above could be two normal predictives."""
    pn = walk_forward(ORDERS[:2], warmup=1)["records"][0]["p_model"]
    pl = walk_forward(ORDERS[:2], warmup=1,
                      base="laplace")["records"][0]["p_model"]
    assert np.abs(pn - pl).max() > 1e-3
