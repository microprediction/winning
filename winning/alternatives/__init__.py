"""Reference implementations of the strongest alternative evaluators.

These are the methods the shared-field engine is benchmarked against,
shipped so the comparison can be rerun and so that each construction
is available to anyone who prefers it. Every function's docstring
says what the engine does differently and when the alternative is the
right choice. Measured comparisons: research/alternatives/ in the
repository and the alternatives page of the documentation site.

reduced_rank_representation
    The Marsaglia / Genz-Bretz reduced-rank rectangle representation
    of ONE winner's probability, ready for mvtnorm::lpRR or any
    rectangle solver. Pure numpy.

cdf_gradient_shares
    All N winner probabilities from one factor-state GHK rectangle
    estimator and one reverse-mode gradient per grid point (requires
    jax). Linear in N; the strongest GHK-family all-share evaluator
    we know of, and still dominated by the shared field within the
    factor grammar because the field integrates the idiosyncratic
    dimensions analytically.
"""
from .reprs import (reduced_rank_representation,
                    per_winner_reduced_rank_shares)
from .cdf_grad import cdf_gradient_shares

__all__ = ["reduced_rank_representation",
           "per_winner_reduced_rank_shares", "cdf_gradient_shares"]
