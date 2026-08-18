from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inference import AbilityCalibrator


def _interp_price_and_slope_1d(cal: AbilityCalibrator, mu: float) -> Tuple[float, float]:
    """Interpolate price and d price / d mu from cached 1D curve."""
    if not cal.lookup_curve_1d_prices:
        raise ValueError(
            "AbilityCalibrator has no 1D lookup curve. Run solve_from_prices or rebuild curves first."
        )
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
    """Interpolate price and slope across location and scale using cached 2D curves; falls back to 1D."""
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
class ConditionSpec:
    cond_id: str
    calibrator: AbilityCalibrator
    item_ids: List[str]
    prices: np.ndarray
    scales: Optional[np.ndarray] = None

    # speed helper
    index: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.index = {hid: i for i, hid in enumerate(self.item_ids)}
        self.prices = np.asarray(self.prices, dtype=float)
        if self.scales is not None:
            self.scales = np.asarray(self.scales, dtype=float)


@dataclass
class MultiRayGlobalCalibrator:
    item_ids: List[str]
    dim: int = 2
    conditions: List[ConditionSpec] = field(default_factory=list)

    # parameters
    Z: Dict[str, np.ndarray] = field(default_factory=dict)  # item -> (dim,)
    V: Dict[str, np.ndarray] = field(default_factory=dict)  # cond_id -> (dim,)
    beta: Dict[str, float] = field(default_factory=dict)  # cond_id -> float

    # regularization
    l2_z: float = 1e-6
    l2_v: float = 1e-6

    # steps
    step_beta: float = 0.3
    step_v: float = 0.3
    step_z: float = 0.3

    # iteration control / safety
    slope_floor: float = 1e-10
    random_state: Optional[int] = None

    # internal maps for speed
    cond_index: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        rng = np.random.default_rng(self.random_state)
        if not self.Z:
            for hid in self.item_ids:
                self.Z[hid] = 0.01 * rng.standard_normal(self.dim)
        self._rebuild_cond_index()

    def _rebuild_cond_index(self) -> None:
        self.cond_index = {c.cond_id: k for k, c in enumerate(self.conditions)}

    def add_condition(
        self,
        cond_id: str,
        calibrator: AbilityCalibrator,
        item_ids: Sequence[str],
        prices: Sequence[float],
        scales: Optional[Sequence[float]] = None,
    ) -> None:
        prices_arr = np.asarray(prices, dtype=float)
        scales_arr = None if scales is None else np.asarray(scales, dtype=float)
        if (calibrator.lookup_curve_1d_prices is None) and (not calibrator.lookup_curves_2d_prices):
            calibrator.solve_from_prices(prices_arr)
        spec = ConditionSpec(
            cond_id=cond_id,
            calibrator=calibrator,
            item_ids=list(item_ids),
            prices=prices_arr,
            scales=scales_arr,
        )
        self.conditions.append(spec)
        # initialize parameters for this condition if absent
        if cond_id not in self.V:
            base_seed = 1469598103934665603  # large-ish offset (FNV basis)
            rs = 0 if self.random_state is None else int(self.random_state)
            seed = (rs + (hash(cond_id) & 0xFFFFFFFF)) ^ base_seed
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim)
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                v = v / nrm
            self.V[cond_id] = v
        if cond_id not in self.beta:
            self.beta[cond_id] = 0.0
        self._rebuild_cond_index()

    def ability(self, cond_id: str, item_id: str) -> float:
        return float(self.beta[cond_id] + float(np.dot(self.V[cond_id], self.Z[item_id])))

    def _predict_and_slopes_for_condition(self, c_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.conditions[c_idx]
        cal = spec.calibrator
        m = len(spec.item_ids)
        p_hat = np.zeros(m, dtype=float)
        slopes = np.zeros(m, dtype=float)
        for k, hid in enumerate(spec.item_ids):
            mu = float(self.ability(spec.cond_id, hid))
            if spec.scales is not None and len(cal.lookup_curves_2d_prices) > 0:
                p, dp = _interp_price_and_slope_2d(cal, mu, float(spec.scales[k]))
            else:
                p, dp = _interp_price_and_slope_1d(cal, mu)
            p_hat[k] = p
            slopes[k] = dp
        return p_hat, slopes

    def rebuild_all_curves(self) -> None:
        for spec in self.conditions:
            mu_r = [self.ability(spec.cond_id, hid) for hid in spec.item_ids]
            if spec.scales is not None and len(mu_r) == len(spec.scales):
                spec.calibrator.rebuild_curves_from_field_2d(mu_r, spec.scales)
            else:
                spec.calibrator.rebuild_curves_from_field_1d(mu_r)

    def apply_gauge_fix(self) -> None:
        # Center embeddings
        if not self.Z:
            return
        Z_stack = np.stack([self.Z[hid] for hid in self.item_ids], axis=0)
        mean_z = np.mean(Z_stack, axis=0)
        for hid in self.item_ids:
            self.Z[hid] = self.Z[hid] - mean_z
        # Normalize each V to unit norm; Z absorbs scaling
        for cid in list(self.V.keys()):
            v = self.V[cid]
            nrm = float(np.linalg.norm(v))
            if nrm > 0:
                self.V[cid] = v / nrm

    def fit_inner(self, num_iters: int) -> None:
        for _ in range(num_iters):
            # Precompute per-condition predictions for reuse in item updates
            cond_p_hat: List[np.ndarray] = []
            cond_slopes: List[np.ndarray] = []
            cond_errors: List[np.ndarray] = []
            for j in range(len(self.conditions)):
                p_hat, slopes = self._predict_and_slopes_for_condition(j)
                e = p_hat - self.conditions[j].prices
                cond_p_hat.append(p_hat)
                cond_slopes.append(slopes)
                cond_errors.append(e)
            # (A) Update condition parameters (beta_j, V_j) with Z fixed
            for j, spec in enumerate(self.conditions):
                p_hat = cond_p_hat[j]
                slopes = cond_slopes[j]
                e = cond_errors[j]
                # clamp slopes
                slopes_safe = slopes.copy()
                mask = np.abs(slopes_safe) < self.slope_floor
                # if dp is exactly zero, treat as positive floor
                slopes_safe[mask] = self.slope_floor
                # targets for delta ability
                y = -e / slopes_safe
                m = len(spec.item_ids)
                X = np.zeros((m, 1 + self.dim), dtype=float)
                X[:, 0] = 1.0
                for k, hid in enumerate(spec.item_ids):
                    X[k, 1:] = self.Z[hid]
                XtX = X.T @ X
                # ridge on v-part only
                XtX[1:, 1:] += self.l2_v * np.eye(self.dim, dtype=float)
                Xty = X.T @ y
                try:
                    w = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
                self.beta[spec.cond_id] += self.step_beta * float(w[0])
                self.V[spec.cond_id] = self.V[spec.cond_id] + self.step_v * np.asarray(
                    w[1:], dtype=float
                )
                # normalize V after update
                v = self.V[spec.cond_id]
                nrm = float(np.linalg.norm(v))
                if nrm > 0:
                    self.V[spec.cond_id] = v / nrm
            # (B) Update item embeddings Z_i with (beta, V) fixed
            for hid in self.item_ids:
                rows = []
                ys = []
                for j, spec in enumerate(self.conditions):
                    if hid not in spec.index:
                        continue
                    k = spec.index[hid]
                    slope_k = cond_slopes[j][k]
                    # clamp slope
                    if abs(slope_k) < self.slope_floor:
                        slope_k = self.slope_floor if slope_k >= 0 else -self.slope_floor
                    e_k = cond_p_hat[j][k] - spec.prices[k]
                    yk = -e_k / slope_k
                    rows.append(self.V[spec.cond_id])
                    ys.append(yk)
                if not rows:
                    continue
                M = np.stack(rows, axis=0)
                y_vec = np.array(ys, dtype=float)
                MtM = M.T @ M + self.l2_z * np.eye(self.dim, dtype=float)
                Mty = M.T @ y_vec
                try:
                    dz = np.linalg.solve(MtM, Mty)
                except np.linalg.LinAlgError:
                    dz = np.linalg.lstsq(MtM, Mty, rcond=None)[0]
                self.Z[hid] = self.Z[hid] + self.step_z * dz

    def fit_with_rebuild(self, num_outer_iters: int = 3, num_inner_iters: int = 10) -> None:
        for _ in range(num_outer_iters):
            self.rebuild_all_curves()
            self.fit_inner(num_inner_iters)
            self.apply_gauge_fix()

    def predict_condition(self, cond_id: str) -> np.ndarray:
        c_idx = self.cond_index[cond_id]
        p_hat, _ = self._predict_and_slopes_for_condition(c_idx)
        return p_hat
