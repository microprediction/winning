import numpy as np

from winning.thurstone import AbilityCalibrator


def test_roundtrip_ability_prices(base):
    cal = AbilityCalibrator(base, n_iter=5)
    true_ability = np.array([-4.0, -1.0, 0.0, +0.7, +2.0], dtype=float)
    prices = cal.state_prices_from_ability(true_ability.tolist())
    # Solve inverse (prices -> ability units directly)
    est = np.array(cal.solve_from_prices(prices), dtype=float)
    # Remove translation by median-centering
    true_c = true_ability - np.median(true_ability)
    est_c = est - np.median(est)
    # 1) rank order must match
    assert list(np.argsort(true_c)) == list(np.argsort(est_c))
    # 2) correlation must be high
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    assert corr > 0.985
    # Optional loose absolute guard
    assert float(np.max(np.abs(true_c - est_c))) < 3.0
