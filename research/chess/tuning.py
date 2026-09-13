"""Select a hyperparameter and refuse to do it silently.

Written after exp41, where every chess experiment in this directory
tuned Glicko-2's `tau` on validation, reported that as evidence of a
fair comparison, and was wrong: `tau` is inert in that implementation.
Sweeping 0.02 to 1.0 returns bit-identical held-out loss, so `min` was
resolving a seven-way tie by taking the first grid entry. Four
experiments repeated it, the claim "neither side is given a free
parameter the other lacks" went into a manuscript, and the competitor
ran at library defaults while our side had a knob that worked. The
correction cost the published headline a third and the estimator
column three fifths.

Nothing raised, because nothing was wrong in a way `min` can see. The
same family as passing a rank array where an order is wanted: no
error, plausible output, wrong conclusion.

`select` makes the three failure modes visible at the call site.

  INERT     the spread of scores across the whole grid is below
            `tol`, so the parameter does not move the metric and
            "tuned on validation" would be a false claim. This is the
            one that cost us.
  EDGE      the winner sits at a grid end, so a better value may lie
            outside and the grid should be extended -- unless the
            score has saturated, which is reported separately because
            an edge on a converged curve is benign (exp41's `period`
            is the example: 0.6203 / 0.6200 / 0.6199 across three
            decades, because past a point the parameter is switched
            off entirely and larger values cannot do more).
  ok        interior, and the parameter demonstrably matters.

Usage:

    scores = {g: run(g).mean() for g in GRID}
    best, report = select(scores, "period")
    print(report)           # or: raise on report.inert in a gate

Deliberately not raising by default. Some parameters are legitimately
inert in some regimes and the right response is to say so in the
write-up, not to crash. What is not acceptable is failing to notice.
"""
from __future__ import annotations
from typing import Dict, NamedTuple, Sequence


class Report(NamedTuple):
    name: str
    best: object
    spread: float
    inert: bool
    at_edge: bool
    saturated: bool

    def __str__(self) -> str:
        bits = [f"{self.name}={self.best!r}"]
        if self.inert:
            bits.append(f"INERT (spread {self.spread:.2e}) -- do not "
                        f"describe this as tuned")
        elif self.at_edge and self.saturated:
            bits.append(f"edge but SATURATED (spread {self.spread:.2e}); "
                        f"benign, the curve has converged")
        elif self.at_edge:
            bits.append("AT GRID EDGE -- extend the grid")
        else:
            bits.append(f"ok (spread {self.spread:.2e})")
        return "  ".join(bits)


def select(scores: Dict[object, float], name: str = "param",
           tol: float = 1e-6, sat_tol: float = 1e-4,
           order: Sequence | None = None):
    """Pick the argmin of `scores` and report how trustworthy that is.

    `scores` maps a grid value to a loss (lower is better). `order`
    gives the grid's natural ordering when the keys are not sortable,
    which is what decides the two end values; it defaults to the
    insertion order of `scores`.

    `tol` is the spread below which the parameter counts as inert.
    `sat_tol` is the gap between the two best-scoring END values below
    which an edge selection counts as saturated rather than unresolved.
    """
    if not scores:
        raise ValueError("no scores")
    keys = list(order) if order is not None else list(scores)
    if len(keys) == 1:
        return keys[0], Report(name, keys[0], 0.0, True, True, False)
    vals = [scores[k] for k in keys]
    best_i = min(range(len(keys)), key=lambda i: vals[i])
    best = keys[best_i]
    spread = max(vals) - min(vals)
    inert = spread < tol
    at_edge = best_i in (0, len(keys) - 1)
    # Saturated: the winning end is indistinguishable from its neighbour,
    # so pushing the grid further cannot help.
    if at_edge and len(keys) >= 2:
        nb = 1 if best_i == 0 else len(keys) - 2
        saturated = abs(vals[best_i] - vals[nb]) < sat_tol
    else:
        saturated = False
    return best, Report(name, best, spread, inert, at_edge, saturated)


def require_live(report: Report) -> None:
    """Raise if a parameter reported as tuned does not move the metric.

    Use in a gate where a fairness claim depends on the tuning being
    real, which is exactly the situation exp41 found we were in.
    """
    if report.inert:
        raise AssertionError(
            f"{report.name} is inert (spread {report.spread:.2e}): "
            f"selecting it is a tie-break, not tuning, and it must not "
            f"be described as tuned in any write-up")
