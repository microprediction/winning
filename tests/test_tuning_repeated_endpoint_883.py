"""Edge flags are read on the unique tuning axis (#883).

With a repeated winning endpoint, (1, 3, 3), argmin is index 1, not the
last raw index, so the public at_edge/at_scalar_limit said interior while
report.at_edge said edge -- and a capped endpoint read as an optimum.
"""
import numpy as np

import winning.ratings.factor_ratings as fr
import winning.ratings.tracker as tr


def test_offset_sweep_repeated_top_endpoint_is_an_edge(monkeypatch):
    # loss falls with the ridge, so the top value wins wherever it appears
    monkeypatch.setattr(fr, "fit_factor_ratings",
                        lambda events, n_entities, n_cov, ridge, **kw: np.asarray(ridge))
    monkeypatch.setattr(fr, "factor_loglik", lambda B, events, base="normal": np.asarray([float(B[1])]))
    out = fr.sweep_offset_ridge([], [0], 2, 2, offset_grid=(1.0, 3.0, 3.0))
    assert out["offset_ridge"] == 3.0
    assert out["at_edge"] and out["at_scalar_limit"] and out["report"].at_edge


def test_offset_sweep_interior_optimum_stays_interior(monkeypatch):
    monkeypatch.setattr(fr, "fit_factor_ratings",
                        lambda events, n_entities, n_cov, ridge, **kw: np.asarray(ridge))
    monkeypatch.setattr(fr, "factor_loglik",
                        lambda B, events, base="normal": np.asarray([-(float(B[1]) - 2.0) ** 2]))
    out = fr.sweep_offset_ridge([], [0], 2, 2, offset_grid=(1.0, 2.0, 2.0, 3.0))
    assert out["offset_ridge"] == 2.0
    assert not out["at_edge"] and not out["at_scalar_limit"] and not out["report"].at_edge


class _FakeTracker:
    def __init__(self, rho=0.0, beta2=1.0, **kw):
        self.evidence = rho + 0.01 * beta2

    def observe(self, *args, **kwargs):
        pass


def test_block_rho_repeated_endpoint_is_an_edge(monkeypatch):
    monkeypatch.setattr(tr, "AbilityTracker", _FakeTracker)
    out = tr.tune_block_rho([{"runners": ["a", "b"], "winner": 0}], rho_grid=(0.0, 0.2, 0.2))
    assert out["rho"] == 0.2 and out["at_edge"] and out["report"].at_edge


def test_block_rho_and_beta2_repeated_endpoint_is_an_edge(monkeypatch):
    monkeypatch.setattr(tr, "AbilityTracker", _FakeTracker)
    out = tr.tune_block_rho([{"runners": ["a", "b"], "winner": 0}],
                            rho_grid=(0.0, 0.2, 0.2), beta2_grid=(1.0, 2.0, 2.0))
    assert out["rho"] == 0.2 and out["at_edge"]
