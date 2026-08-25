"""winning: dealing with races, correlated or not.

The front door, min-wins convention (lower mu is better, as in race
times; negate utilities, or use winning.probit for max-wins semantics):

    race_probabilities(mu, V=, D=, base=, temperature=)
    calibrate_abilities(p, ...)      inverse: abilities from win probabilities
    removal_shares(mu, ...)          P(j wins | i removed), all pairs, one field
    tie_densities(mu, ...)           photo-finish weights (circuit conductances)

winning.probit speaks the factor-probit literature's language (max-wins
utilities, shares, supplied covariances). winning.factor holds the
paper-faithful kernels; winning.thurstone the density-agnostic engine.
calibrate_factors (outer estimation) is reserved for a future release.
"""

from .factor import (  # noqa: F401
    calibrate_abilities,
    race_probabilities,
    removal_shares,
    tie_densities,
)
from . import probit  # noqa: F401

__version__ = "1.1.1"
