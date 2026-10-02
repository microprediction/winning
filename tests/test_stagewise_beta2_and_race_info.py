"""update_ranking with per-player beta2 (#138) and update_race's
logZ_outcome on every outcome branch (#144)."""

import numpy as np
import pytest

from winning.ratings.market import update_race
from winning.ratings.nway import update_ranking, update_winner


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2]])
def test_a_constant_vector_equals_the_scalar(order):
    m, v = np.array([0.2, -0.1, 0.4]), np.array([1.0, 0.5, 2.0])
    a = update_ranking(m, v, order, beta2=0.7)
    b = update_ranking(m, v, order, beta2=np.full(3, 0.7))
    assert np.allclose(a[0], b[0], atol=1e-14)
    assert np.allclose(a[1], b[1], atol=1e-14)


def test_heterogeneous_beta2_is_the_stagewise_composition():
    m, v = np.zeros(3), np.ones(3)
    b2 = np.array([0.5, 1.0, 2.0])
    got_m, got_v = update_ranking(m, v, [0, 1, 2], beta2=b2)
    # by hand: stage 1 on the full field, stage 2 on runners [1, 2]
    m1, v1, _ = update_winner(m, v, 0, beta2=b2)
    m2, v2, _ = update_winner(m1[[1, 2]], v1[[1, 2]], 0, beta2=b2[[1, 2]])
    want_m = m1.copy(); want_m[[1, 2]] = m2
    want_v = v1.copy(); want_v[[1, 2]] = v2
    assert np.allclose(got_m, want_m, atol=1e-14)
    assert np.allclose(got_v, want_v, atol=1e-14)
    # and the heterogeneity moves it
    assert np.abs(got_m - update_ranking(m, v, [0, 1, 2])[0]).max() > 1e-3


@pytest.mark.parametrize("kw,want", [
    ({"order": [0, 1, 2]}, -np.log(6.0)),
    # a partial order is over the runners who finished (the rest are
    # marginalized, update_ranking_exact docstring): 2 beat 0, P = 1/2
    ({"order": [2, 0]}, -np.log(2.0)),
    ({"order": [1, 0, 2], "V": [[0.3], [-0.2], [0.1]]}, None),
    ({"winner": 1}, -np.log(3.0)),
    ({"winner": 1, "V": [[0.3], [-0.2], [0.1]]}, None),
])
def test_every_outcome_branch_reports_its_evidence(kw, want):
    _, _, info = update_race(np.zeros(3), np.ones(3), **kw)
    assert "logZ_outcome" in info, (kw, info)
    assert np.isfinite(info["logZ_outcome"])
    if want is not None:
        assert abs(info["logZ_outcome"] - want) < 2e-4, (kw, info)
