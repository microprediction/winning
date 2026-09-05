from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inference import AbilityCalibrator


def _interp_price_and_slope_1d(cal: AbilityCalibrator, mu: float) -> Tuple[float, float]:
    """Interpolate price and d price / d mu from cached 1D curve."""
    if not cal.lookup_curve_1d_prices:
        raise ValueError("AbilityCalibrator has no 1D lookup curve. Run solve_from_prices first.")
    locs = cal.lookup_curve_1d_prices["locs"]
    prices = cal.lookup_curve_1d_prices["prices"]
    dprices = np.gradient(prices, locs)
    mu_c = float(np.clip(mu, locs.min(), locs.max()))
    p = float(np.interp(mu_c, locs, prices))
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    dp = float(np.interp(mu_c, locs, dprices))
    return p, dp


def _interp_price_and_slope_2d(
    cal: AbilityCalibrator, mu: float, scale: float
) -> Tuple[float, float]:
    """Interpolate price and slope across location and scale using cached 2D curves."""
    if not cal.lookup_curves_2d_prices:
        return _interp_price_and_slope_1d(cal, mu)
    scales = sorted(cal.lookup_curves_2d_prices.keys())
    s_arr = np.array(scales, dtype=float)
    if scale in cal.lookup_curves_2d_prices:
        locs, prices = cal.lookup_curves_2d_prices[float(scale)]
        dprices = np.gradient(prices, locs)
        mu_c = float(np.clip(mu, locs.min(), locs.max()))
        p = float(np.interp(mu_c, locs, prices))
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        dp = float(np.interp(mu_c, locs, dprices))
        return p, dp
    idx = int(np.searchsorted(s_arr, scale))
    if idx <= 0:
        s1, s2 = s_arr[0], s_arr[min(1, len(s_arr) - 1)]
    elif idx >= len(s_arr):
        s1, s2 = s_arr[-2], s_arr[-1]
    else:
        s1, s2 = s_arr[idx - 1], s_arr[idx]
    w = 0.0 if s2 == s1 else (float(scale) - s1) / (s2 - s1)
    locs1, prices1 = cal.lookup_curves_2d_prices[float(s1)]
    dprices1 = np.gradient(prices1, locs1)
    mu1 = float(np.clip(mu, locs1.min(), locs1.max()))
    p1 = float(np.interp(mu1, locs1, prices1))
    dp1 = float(np.interp(mu1, locs1, dprices1))
    locs2, prices2 = cal.lookup_curves_2d_prices[float(s2)]
    dprices2 = np.gradient(prices2, locs2)
    mu2 = float(np.clip(mu, locs2.min(), locs2.max()))
    p2 = float(np.interp(mu2, locs2, prices2))
    dp2 = float(np.interp(mu2, locs2, dprices2))
    p = (1.0 - w) * p1 + w * p2
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    dp = (1.0 - w) * dp1 + w * dp2
    return p, dp


@dataclass
class RaceSpec:
    calibrator: AbilityCalibrator
    horse_ids: List[str]
    prices: np.ndarray
    scales: Optional[np.ndarray] = None


@dataclass
class GlobalAbilityCalibrator:
    horse_ids: List[str]
    races: List[RaceSpec] = field(default_factory=list)
    theta: Dict[str, float] = field(default_factory=dict)
    biases: List[float] = field(default_factory=list)
    l2: float = 1e-8
    step_bias: float = 0.3
    step_theta: float = 0.3

    def __post_init__(self):
        if not self.theta:
            self.theta = {hid: 0.0 for hid in self.horse_ids}

    def add_race(
        self,
        calibrator: AbilityCalibrator,
        horse_ids: Sequence[str],
        prices: Sequence[float],
        scales: Optional[Sequence[float]] = None,
    ) -> None:
        prices_arr = np.asarray(prices, dtype=float)
        scales_arr = None if scales is None else np.asarray(scales, dtype=float)
        if (calibrator.lookup_curve_1d_prices is None) and (not calibrator.lookup_curves_2d_prices):
            calibrator.solve_from_prices(prices_arr)
        self.races.append(
            RaceSpec(
                calibrator=calibrator,
                horse_ids=list(horse_ids),
                prices=prices_arr,
                scales=scales_arr,
            )
        )
        self.biases.append(0.0)

    def _predict_and_slopes_for_race(self, r_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.races[r_idx]
        cal = spec.calibrator
        p_hat = np.zeros(len(spec.horse_ids), dtype=float)
        slopes = np.zeros(len(spec.horse_ids), dtype=float)
        for i, hid in enumerate(spec.horse_ids):
            mu = float(self.theta[hid] + self.biases[r_idx])
            if spec.scales is not None and len(cal.lookup_curves_2d_prices) > 0:
                p, dp = _interp_price_and_slope_2d(cal, mu, float(spec.scales[i]))
            else:
                p, dp = _interp_price_and_slope_1d(cal, mu)
            p_hat[i] = p
            slopes[i] = dp
        return p_hat, slopes

    def fit(self, num_iters: int = 25) -> None:
        for _ in range(num_iters):
            for r in range(len(self.races)):
                spec = self.races[r]
                p_hat, slopes = self._predict_and_slopes_for_race(r)
                e = p_hat - spec.prices
                denom = float(np.dot(slopes, slopes) + self.l2)
                if denom > 0:
                    delta = -float(np.dot(slopes, e)) / denom
                    self.biases[r] += self.step_bias * delta
            for hid in self.horse_ids:
                num = 0.0
                den = self.l2
                for r, spec in enumerate(self.races):
                    if hid not in spec.horse_ids:
                        continue
                    i = spec.horse_ids.index(hid)
                    cal = spec.calibrator
                    mu = float(self.theta[hid] + self.biases[r])
                    if spec.scales is not None and len(cal.lookup_curves_2d_prices) > 0:
                        p, dp = _interp_price_and_slope_2d(cal, mu, float(spec.scales[i]))
                    else:
                        p, dp = _interp_price_and_slope_1d(cal, mu)
                    e = p - float(spec.prices[i])
                    num += dp * e
                    den += dp * dp
                if den > 0:
                    self.theta[hid] -= self.step_theta * float(num / den)

    def predict_race(self, r_idx: int) -> np.ndarray:
        p_hat, _ = self._predict_and_slopes_for_race(r_idx)
        return p_hat

    def rebuild_all_curves(self) -> None:
        for r, spec in enumerate(self.races):
            mu_r = [float(self.theta[hid] + self.biases[r]) for hid in spec.horse_ids]
            if spec.scales is not None and len(mu_r) == len(spec.scales):
                spec.calibrator.rebuild_curves_from_field_2d(mu_r, spec.scales)
            else:
                spec.calibrator.rebuild_curves_from_field_1d(mu_r)

    def fit_with_rebuild(self, num_outer_iters: int = 3, num_inner_iters: int = 10) -> None:
        for _ in range(num_outer_iters):
            self.rebuild_all_curves()
            self.fit(num_inner_iters)

    # --- Variants that hold race biases fixed (theta-only fitting) ---
    def fit_theta_only(self, num_iters: int = 25) -> None:
        for _ in range(num_iters):
            for hid in self.horse_ids:
                num = 0.0
                den = self.l2
                for r, spec in enumerate(self.races):
                    if hid not in spec.horse_ids:
                        continue
                    i = spec.horse_ids.index(hid)
                    cal = spec.calibrator
                    mu = float(self.theta[hid] + self.biases[r])
                    if spec.scales is not None and len(cal.lookup_curves_2d_prices) > 0:
                        p, dp = _interp_price_and_slope_2d(cal, mu, float(spec.scales[i]))
                    else:
                        p, dp = _interp_price_and_slope_1d(cal, mu)
                    e = p - float(spec.prices[i])
                    num += dp * e
                    den += dp * dp
                if den > 0:
                    self.theta[hid] -= self.step_theta * float(num / den)

    def fit_with_rebuild_theta_only(
        self, num_outer_iters: int = 3, num_inner_iters: int = 10
    ) -> None:
        for _ in range(num_outer_iters):
            self.rebuild_all_curves()
            self.fit_theta_only(num_inner_iters)
