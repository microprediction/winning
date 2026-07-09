"""Scoring rules for prequential rating-system evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

_CLIP = 1e-12


@dataclass
class Metrics:
    """Accumulates per-event scores; report with summary()."""

    log_losses: List[float] = field(default_factory=list)
    briers: List[float] = field(default_factory=list)
    hits: List[float] = field(default_factory=list)
    taus: List[float] = field(default_factory=list)
    truth_gaps: List[float] = field(default_factory=list)
    cal_probs: List[float] = field(default_factory=list)  # every contestant win prob
    cal_wins: List[float] = field(default_factory=list)  # ... and whether they won
    pits: List[float] = field(default_factory=list)  # randomized rank-PIT values
    seconds: float = 0.0

    def score_event(
        self,
        probs: Sequence[float],
        ranks: Sequence[int],
        pred_order_mus: Sequence[float],
        truth: Sequence[float] = None,
    ) -> None:
        n = len(probs)
        winner = min(range(n), key=lambda i: ranks[i])
        p = [min(max(float(q), _CLIP), 1.0) for q in probs]
        total = sum(p)
        p = [q / total for q in p]

        self.log_losses.append(-math.log(p[winner]))
        self.briers.append(sum((p[i] - (1.0 if i == winner else 0.0)) ** 2 for i in range(n)))
        best = max(p)
        top = [i for i in range(n) if p[i] == best]
        self.hits.append(1.0 / len(top) if winner in top else 0.0)
        self.taus.append(kendall_tau(pred_order_mus, ranks))
        self.cal_probs.extend(p)
        self.cal_wins.extend(1.0 if i == winner else 0.0 for i in range(n))
        if truth is not None:
            self.truth_gaps.append(
                sum(abs(p[i] - truth[i]) for i in range(n)) / 2.0  # total variation
            )

    def summary(self) -> dict:
        def _mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        return {
            "events": len(self.log_losses),
            "log_loss": _mean(self.log_losses),
            "brier": _mean(self.briers),
            "accuracy": _mean(self.hits),
            "kendall_tau": _mean(self.taus),
            "ece": expected_calibration_error(self.cal_probs, self.cal_wins),
            "rank_pit_ks": ks_from_uniform(self.pits) if self.pits else None,
            "tv_vs_oracle": _mean(self.truth_gaps) if self.truth_gaps else None,
            "seconds": self.seconds,
        }


def ks_from_uniform(us: Sequence[float]) -> float:
    """Kolmogorov distance of the rank-PIT sample from Uniform(0,1): 0 means
    the predicted finish-position distributions match the realized ones —
    i.e. performance variation is modeled correctly, not just the mean."""
    n = len(us)
    xs = sorted(us)
    return max(
        max(abs((i + 1) / n - x), abs(x - i / n)) for i, x in enumerate(xs)
    )


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[float], bins: int = 20
) -> float:
    """Equal-count-binned |mean predicted - mean observed|, weighted by bin size.

    Measures whether the spread of predicted win probabilities matches reality:
    a system that misjudges performance variation is systematically over- or
    under-confident and scores high here.
    """
    n = len(probs)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: probs[i])
    total = 0.0
    for b in range(bins):
        lo, hi = (b * n) // bins, ((b + 1) * n) // bins
        if hi <= lo:
            continue
        idx = order[lo:hi]
        p_bar = sum(probs[i] for i in idx) / len(idx)
        y_bar = sum(outcomes[i] for i in idx) / len(idx)
        total += abs(p_bar - y_bar) * len(idx)
    return total / n


def kendall_tau(pred_mus: Sequence[float], ranks: Sequence[int]) -> float:
    """Kendall tau between the order implied by pred_mus (higher = better,
    i.e. predicted to finish earlier) and observed 1-based ranks.

    Tie policy (deliberate, between tau-a and tau-b): tied pairs — in either
    the prediction or the outcome — count in the denominator with zero credit,
    so a system that cannot separate contestants scores 0 on those pairs
    rather than being excused for them."""
    n = len(ranks)
    num = 0
    den = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = pred_mus[i] - pred_mus[j]
            b = ranks[j] - ranks[i]  # positive when i finished ahead
            if a == 0 or b == 0:
                den += 1
                continue
            num += 1 if (a > 0) == (b > 0) else -1
            den += 1
    return num / den if den else 0.0
