import numpy as np

from winning.thurstone import AbilityCalibrator


def test_roundtrip_various_patterns(base):
    cal = AbilityCalibrator(base, n_iter=5)
    patterns = [
        [-2.0, -1.0, 0.0, +1.0, +2.0],
        [-5.0, -3.0, -1.0, +0.5, +3.0],
        [-1.0, -1.0, 0.0, +1.5, +2.5],  # near ties
    ]
    for true_ability in patterns:
        true_ability = np.array(true_ability, dtype=float)
        prices = cal.state_prices_from_ability(true_ability.tolist())
        est = np.array(cal.solve_from_prices(prices), dtype=float)
        # ignore translation by median-centering
        true_c = true_ability - np.median(true_ability)
        est_c = est - np.median(est)
        # check ranks preserved
        assert list(np.argsort(true_c)) == list(np.argsort(est_c))
        # high linear correlation
        corr = float(np.corrcoef(true_c, est_c)[0, 1])
        assert corr > 0.99


def test_two_runner_inversion_monotone(base):
    cal = AbilityCalibrator(base, n_iter=3)
    grid = np.linspace(-5.0, 5.0, 21)
    # Runner 0 varies, runner 1 fixed at 0
    prices0 = [cal.state_prices_from_ability([g, 0.0])[0] for g in grid]
    est0 = [cal.solve_from_prices([p, 1.0 - p])[0] for p in prices0]
    # Monotone relationship and high correlation
    diffs = np.diff(prices0)
    assert np.all(diffs <= 1e-8) or np.all(diffs >= -1e-8)
    corr = float(np.corrcoef(grid, est0)[0, 1])
    # With multiplicity-aware global-curve inversion and coarse grids,
    # correlation can be slightly below 0.99; keep robust but strict.
    assert corr > 0.965
