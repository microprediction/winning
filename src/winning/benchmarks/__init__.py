"""Prequential benchmarking of rating systems: predict each event before
observing it, score, update. Requires `pip install winning[benchmarks]` for
the third-party comparators."""

from .events import Event
from .forward_chain import evaluate
from .metrics import Metrics

__all__ = ["Event", "evaluate", "Metrics"]
