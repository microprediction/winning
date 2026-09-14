"""Hyperparameter selection that reports whether the tuning was real.

Selecting the argmin of a validation sweep is not the same as tuning.
If the parameter does not move the metric, the sweep returns a tie and
``min`` resolves it by taking whichever grid entry came first. The run
looks successful, a value is reported as chosen, and the comparison it
was meant to make fair is not.

That is not hypothetical. Across four experiments in
``research/chess``, Glicko-2's ``tau`` was swept on a validation split
and the selected value quoted as evidence that neither side of a
comparison had a free parameter the other lacked. In that
implementation ``tau`` enters only the volatility iteration, the
volatility barely moves over a single month, and sweeping 0.02 to 1.0
returned bit-identical held-out loss. Every grid began at 0.3, so the
tie-break picked 0.3 every time and nothing looked lower. The
parameters that governed the competitor's behaviour were never swept
at all, and correcting it cost the published result a third of its
headline and three fifths of one column.

``select`` returns the argmin together with the three ways a sweep can
mislead.

    INERT      the spread of scores across the grid is below ``tol``.
               The parameter does not move the metric, so the choice
               is a tie-break and must not be described as tuned.
    EDGE       the winner sits at a grid end, so a better value may
               lie outside it and the grid wants extending.
    SATURATED  the winner sits at an end but is indistinguishable from
               its neighbour. The curve has converged and the edge is
               benign. A recency parameter large enough to switch off
               time decay behaves this way: further increases cannot
               do anything.

Worked example, the four cases as they actually arose:

    >>> select({0.02: 0.6273, 0.3: 0.6273, 1.0: 0.6273}, "tau").report
    tau=0.02  INERT ...
    >>> select({200.0: 0.6203, 1500.0: 0.6200, 20000.0: 0.6199},
    ...        "period").report
    period=20000.0  edge but SATURATED ...

Nothing raises by default. A parameter can be legitimately inert in a
given regime, and the right response is to say so in the write-up
rather than to crash. Use ``require_live`` where a fairness claim
depends on the tuning having been real, which is the case a sweep
across two competing systems always is.
"""
from __future__ import annotations

from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple


class TuningReport(NamedTuple):
    """How much to trust a selected hyperparameter."""

    name: str
    best: object
    spread: float
    inert: bool
    at_edge: bool
    saturated: bool

    @property
    def trustworthy(self) -> bool:
        """True when the parameter moves the metric and the grid holds it."""
        return not self.inert and (self.saturated or not self.at_edge)

    def __str__(self) -> str:
        head = f"{self.name}={self.best!r}"
        if self.inert:
            return (f"{head}  INERT (spread {self.spread:.2e}): the choice "
                    f"is a tie-break, not tuning")
        if self.at_edge and self.saturated:
            return (f"{head}  edge but SATURATED (spread {self.spread:.2e}): "
                    f"the curve has converged and the edge is benign")
        if self.at_edge:
            return f"{head}  AT GRID EDGE: extend the grid"
        return f"{head}  ok (spread {self.spread:.2e})"


def select(scores: Mapping[object, float], name: str = "param",
           tol: float = 1e-6, sat_tol: float = 1e-4,
           order: Optional[Sequence] = None) -> Tuple[object, TuningReport]:
    """Pick the lowest-scoring grid value and report how it was reached.

    ``scores`` maps each grid value to a validation loss, lower being
    better. ``order`` supplies the grid's natural ordering when the
    keys do not sort meaningfully, and decides which two values count
    as the ends; it defaults to the insertion order of ``scores``.

    ``tol`` is the spread below which the parameter counts as inert.
    ``sat_tol`` is the gap between a winning end value and its
    neighbour below which an edge counts as saturated.

    Returns ``(best, report)``.
    """
    if not scores:
        raise ValueError("select needs a non-empty mapping of scores")
    keys = list(order) if order is not None else list(scores)
    missing = [k for k in keys if k not in scores]
    if missing:
        raise KeyError(f"order lists values absent from scores: {missing}")
    values = [float(scores[k]) for k in keys]
    best_i = min(range(len(keys)), key=lambda i: values[i])
    best = keys[best_i]
    spread = max(values) - min(values)
    if len(keys) == 1:
        return best, TuningReport(name, best, 0.0, True, True, False)
    at_edge = best_i in (0, len(keys) - 1)
    saturated = False
    if at_edge:
        neighbour = 1 if best_i == 0 else len(keys) - 2
        saturated = abs(values[best_i] - values[neighbour]) < sat_tol
    return best, TuningReport(name, best, spread, spread < tol,
                              at_edge, saturated)


def select_grid(scores: Mapping[Tuple, float], names: Sequence[str],
                tol: float = 1e-6, sat_tol: float = 1e-4
                ) -> Tuple[Tuple, Dict[str, TuningReport]]:
    """Select over a product grid and report each axis separately.

    ``scores`` maps a tuple of parameter values to a validation loss.
    Reporting the tuple against the ends of the flattened list hides an
    axis sitting at its own end, so each axis is profiled by the best
    score achievable at each of its values and reported on its own.

    Returns ``(best_tuple, {axis_name: report})``.
    """
    if not scores:
        raise ValueError("select_grid needs a non-empty mapping of scores")
    combos = list(scores)
    width = len(combos[0])
    if len(names) != width:
        raise ValueError(f"{len(names)} names for {width} axes")
    best = min(combos, key=lambda c: scores[c])
    reports = {}
    for axis, nm in enumerate(names):
        values = sorted({c[axis] for c in combos})
        profile = {v: min(sc for c, sc in scores.items() if c[axis] == v)
                   for v in values}
        _, reports[nm] = select(profile, nm, tol=tol, sat_tol=sat_tol,
                                order=values)
    return best, reports


def require_live(report: TuningReport) -> None:
    """Raise when a parameter described as tuned does not move the metric.

    Call this wherever a claim of comparable tuning effort rests on the
    sweep having done something.
    """
    if report.inert:
        raise AssertionError(
            f"{report.name} is inert (spread {report.spread:.2e}): selecting "
            f"it is a tie-break, not tuning, and it cannot be reported as a "
            f"tuned parameter")
