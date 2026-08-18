from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .density import Density
from .order_stats import winner_of_many
from .pricing import Race


def _int_centered(offsets: Sequence[float]) -> List[float]:
    finite = [o for o in offsets if np.isfinite(o)]
    if not finite:
        return list(offsets)
    mean_int = int(np.mean(finite))
    return [o - mean_int for o in offsets]


def _divide_offsets(centered_offsets: Sequence[float]) -> float:
    """Choose a divider that maximizes the biggest gap near the center; ensures two non-empty groups."""
    srt = sorted(centered_offsets)
    if len(srt) <= 2:
        return float(np.mean(srt))
    gaps = np.diff(srt)
    idx = int(np.argmax(np.abs(gaps)))
    return float(0.5 * (srt[idx] + srt[idx + 1]))


@dataclass
class ClusterSplitter:
    unit_ratio: float = 3.0
    max_depth: int = 3

    def extended_state_prices(self, base: Density, offsets: Sequence[float]) -> List[float]:
        """Offsets may include +/-inf; returns normalized winning probabilities."""
        n = len(offsets)
        if n == 1:
            return [1.0]
        # handle +inf (no chance)
        pos_inf_idx = [i for i, o in enumerate(offsets) if o == float("inf")]
        if pos_inf_idx:
            finite_idx = [i for i in range(n) if i not in pos_inf_idx]
            if not finite_idx:
                return [1.0 / n for _ in range(n)]
            finite_prices = self.extended_state_prices(
                base, _int_centered([offsets[i] for i in finite_idx])
            )
            out = [0.0 for _ in range(n)]
            for j, idx in enumerate(finite_idx):
                out[idx] = finite_prices[j]
            return out
        # handle -inf (certain winners share)
        neg_inf_idx = [i for i, o in enumerate(offsets) if o == float("-inf")]
        if neg_inf_idx:
            p = [0.0 for _ in range(n)]
            share = 1.0 / len(neg_inf_idx)
            for idx in neg_inf_idx:
                p[idx] = share
            return p

        L = base.lattice.L
        W = int(base.approx_support_width())
        centered = _int_centered(offsets)

        # boundaries
        lower_bound = -L + W
        upper_bound = L - W
        hang_left = [i for i, o in enumerate(centered) if o < lower_bound]
        hang_right = [i for i, o in enumerate(centered) if o > upper_bound]
        if (not hang_left) and (not hang_right):
            from .inference import densities_from_offsets, state_prices_from_densities

            dens = densities_from_offsets(base, centered)
            return state_prices_from_densities(dens)

        # stop recursion
        if self.max_depth <= 0:
            for i in hang_right:
                centered[i] = float("inf")
            for i in hang_left:
                centered[i] = float("-inf")
            return self.extended_state_prices(base, centered)

        # split symmetrically
        divider = _divide_offsets(centered)
        left_idx = [i for i, o in enumerate(centered) if o < divider]
        right_idx = [i for i, o in enumerate(centered) if o >= divider]
        if len(left_idx) == 0 or len(right_idx) == 0:
            if len(left_idx) == 0:
                left_idx = [int(np.argmin(centered))]
                right_idx = [i for i in range(n) if i not in left_idx]
            else:
                right_idx = [int(np.argmax(centered))]
                left_idx = [i for i in range(n) if i not in right_idx]

        # group representatives via first-order statistics (race between subgroup winners)
        from .inference import densities_from_offsets, state_prices_from_densities

        dens_left = densities_from_offsets(base, [centered[i] for i in left_idx])
        dens_right = densities_from_offsets(base, [centered[i] for i in right_idx])
        if len(dens_left) == 0 or len(dens_right) == 0:
            # Fallback safety; should be avoided by index balancing above
            dens = densities_from_offsets(base, centered)
            return state_prices_from_densities(dens)
        rep_left, _ = winner_of_many(dens_left)
        rep_right, _ = winner_of_many(dens_right)
        group_prices = Race([rep_left, rep_right]).state_prices()
        left_share = float(group_prices[0])
        right_share = float(group_prices[1])

        # refine inside groups using "weak-as-single" logic:
        # treat the weaker group as a single contestant for the cross-group race (already done via reps),
        # then allocate within the weak group by a race inside that subgroup; for the strong group,
        # use direct detailed prices (no additional recursion) to preserve resolution.
        if left_share <= right_share:
            # left group is weaker
            left_prices_rel = self.__class__(
                self.unit_ratio, self.max_depth - 1
            ).extended_state_prices(base, [centered[i] for i in left_idx])
            from .inference import state_prices_from_densities

            right_prices_rel = state_prices_from_densities(dens_right)
            # normalize detailed prices
            S_r = float(sum(right_prices_rel))
            if S_r > 0:
                right_prices_rel = [pr / S_r for pr in right_prices_rel]
        else:
            # right group is weaker
            right_prices_rel = self.__class__(
                self.unit_ratio, self.max_depth - 1
            ).extended_state_prices(base, [centered[i] for i in right_idx])
            from .inference import state_prices_from_densities

            left_prices_rel = state_prices_from_densities(dens_left)
            S_l = float(sum(left_prices_rel))
            if S_l > 0:
                left_prices_rel = [pl / S_l for pl in left_prices_rel]

        out = [0.0 for _ in range(n)]
        for j, idx in enumerate(left_idx):
            out[idx] = left_share * left_prices_rel[j]
        for j, idx in enumerate(right_idx):
            out[idx] = right_share * right_prices_rel[j]

        S = sum(out)
        if S <= 0:
            raise ValueError("Extended state prices have non-positive total mass.")
        assert 0.999 <= S <= 1.001, f"State prices not normalized in extended offsets; sum={S}"
        return [oi / S for oi in out]
