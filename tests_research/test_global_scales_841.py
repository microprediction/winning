"""#841: GlobalLSCalibrator / GlobalAbilityCalibrator must use per-runner scales."""
import numpy as np
import pytest

from winning.research import (
    AbilityCalibrator,
    Density,
    GlobalAbilityCalibrator,
    GlobalLSCalibrator,
    Race,
    UniformLattice,
)

TRUTH = np.array([-1.5, 0.0, 1.5])
SCALES = np.array([0.35, 1.25, 2.5])
IDS = ["a", "b", "c"]


@pytest.fixture(scope="module")
def setup():
    lat = UniformLattice(L=150, unit=0.2)
    base = Density.skew_normal(lat, loc=0, scale=1, a=0)
    gen = AbilityCalibrator(base, loc_span=5, loc_step=0.1, scale_steps=1)
    dens = [gen.density_for(float(m), float(s)) for m, s in zip(TRUTH, SCALES)]
    return base, np.asarray(Race(dens).state_prices())


def _cal(base):
    return AbilityCalibrator(base, n_iter=3, loc_span=5, loc_step=0.1, scale_steps=1)


def _ls(base, prices, scales):
    cal = _cal(base)
    fit = GlobalLSCalibrator(list(IDS))
    fit.add_race(cal, IDS, prices, scales=scales)
    fit.fit()
    return cal, fit


def test_global_ls_recovers_heteroskedastic_abilities(setup):
    base, prices = setup
    cal, fit = _ls(base, prices, SCALES)
    theta = np.array([fit.theta[h] for h in IDS])
    np.testing.assert_allclose(theta, TRUTH, atol=0.01)
    assert cal.scales is None  # explicit scales do not mutate the calibrator


def test_global_ls_scale_permutation_matters(setup):
    base, prices = setup
    _, f1 = _ls(base, prices, SCALES)
    _, f2 = _ls(base, prices, SCALES[::-1])
    assert np.max(np.abs(f1.races[0].local_locs - f2.races[0].local_locs)) > 0.1


def test_global_ls_without_scales_is_1d_inverse(setup):
    base, prices = setup
    _, fit = _ls(base, prices, None)
    direct = np.array(_cal(base).solve_from_prices(prices), dtype=float)
    np.testing.assert_array_equal(fit.races[0].local_locs, direct)


def test_solve_from_prices_scales_kwarg_matches_set_scales(setup):
    base, prices = setup
    ambient = _cal(base)
    ambient.set_scales(SCALES)
    np.testing.assert_array_equal(
        _cal(base).solve_from_prices(prices, scales=SCALES), ambient.solve_from_prices(prices)
    )


def test_global_ability_plain_fit_uses_scales(setup):
    base, prices = setup
    thetas = []
    for s in (SCALES, SCALES[::-1]):
        g = GlobalAbilityCalibrator(list(IDS))
        g.add_race(_cal(base), IDS, prices, scales=s)
        g.fit(50)
        np.testing.assert_allclose(g.predict_race(0), prices, atol=1e-3)
        thetas.append(np.array([g.theta[h] for h in IDS]))
    assert np.max(np.abs(thetas[0] - thetas[1])) > 0.1


@pytest.mark.parametrize("bad", [[1.0, 1.0], [1.0, 0.0, 1.0], [1.0, np.nan, 1.0]])
@pytest.mark.parametrize("cls", [GlobalLSCalibrator, GlobalAbilityCalibrator])
def test_invalid_scales_rejected(setup, cls, bad):
    base, prices = setup
    with pytest.raises(ValueError):
        cls(list(IDS)).add_race(_cal(base), IDS, prices, scales=bad)
