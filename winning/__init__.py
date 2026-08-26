"""winning: dealing with races, correlated or not.

ONE race, five covariance grammars. Every model here is the same Gaussian
min-race Y = mu + noise; the structures (winning.factor.structures) are
declarative descriptions of the noise covariance, each with an O(N) field
pass: Independent, Factor, Blocks, Nested, Tree.

The front door, min-wins convention (lower mu is better, as in race
times; negate utilities, or use winning.probit for max-wins semantics):

    race_probabilities(mu, structure=, base=, temperature=)   (V=/D= = Factor sugar)
    calibrate_abilities(p, ...)      inverse: abilities from win probabilities
    race_jacobian(mu, ...)           exact dp/dmu           (factor.polish)
    polish_race(p, ..., caps)        nearest race obeying linear constraints
    removal_shares(mu, ...)          P(j wins | i removed), all pairs, one field
    tie_densities(mu, ...)           photo-finish weights (circuit conductances)

Block/nested/tree kernels: winning.factor.blocks (fastrace-accelerated where
available); inversion for blocks via abilities_from_block_race.

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
