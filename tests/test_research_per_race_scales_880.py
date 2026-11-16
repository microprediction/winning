"""Per-race scales on shared research calibrators (#880).

An explicit scales= override was not validated (a wrong length fell back
to the 1-D model in silence, a zero divided by it), a scaled race followed
by an unscaled one left no 1-D curve, and a second scaled race cleared
the 2-D cache down to its own scales, so fitting the first raised.
"""
import numpy as np
import pytest

from winning.research import AbilityCalibrator, Density, GlobalAbilityCalibrator, UniformLattice

LAT = UniformLattice(L=40, unit=0.2)
BASE = Density.skew_normal(LAT, loc=0, scale=1, a=0)
PRICES = np.array([0.5, 0.3, 0.2])
IDS = ["a", "b", "c"]


def _cal():
    return AbilityCalibrator(BASE, n_iter=1, loc_span=2, loc_step=1, scale_steps=1)


@pytest.mark.parametrize("scales", [[1.0, 2.0], [1.0, 0.0, 1.0], [1.0, np.nan, 1.0], [[1.0, 1.0, 1.0]]])
def test_an_explicit_override_is_validated(scales):
    with pytest.raises(ValueError, match="scales"):
        _cal().solve_from_prices(PRICES, scales=scales)


def test_a_scaled_then_unscaled_race_fits():
    c, g = _cal(), GlobalAbilityCalibrator(IDS)
    g.add_race(c, IDS, PRICES, scales=[0.5, 1.0, 1.5])
    g.add_race(c, IDS, PRICES)
    assert c.lookup_curve_1d_prices is not None
    g.fit(1)
    assert all(np.isfinite(g.predict_race(r)).all() for r in range(2))


def test_two_scaled_races_keep_both_scale_sets():
    c, g = _cal(), GlobalAbilityCalibrator(IDS)
    g.add_race(c, IDS, PRICES, scales=[0.35, 1.25, 2.5])
    g.add_race(c, IDS, PRICES, scales=[1.0, 1.0, 1.0])
    assert {0.35, 1.0, 1.25, 2.5} <= set(c.lookup_curves_2d_prices)
    g.fit(1)
    g.rebuild_all_curves()
    assert {0.35, 1.0, 1.25, 2.5} <= set(c.lookup_curves_2d_prices)
    assert all(np.isfinite(g.predict_race(r)).all() for r in range(2))
