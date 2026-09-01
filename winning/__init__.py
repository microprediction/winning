"""winning: dealing with races, correlated or not.

ONE race, five covariance grammars. Every model here is the same Gaussian
min-race Y = mu + noise; the structures (winning.factor.structures) are
declarative descriptions of the noise covariance, each with an O(N) field
pass: Independent, Factor, Blocks, Nested, Tree.

The front door, min-wins convention (lower mu is better, as in race
times; negate utilities, or use winning.probit for max-wins semantics):

    race_probabilities(mu, structure=, base=, temperature=)   (V=/D= = Factor sugar)
    calibrate_abilities(p, ...)      inverse: abilities from win probabilities
    softmax_probabilities(mu, ...)   the Luce/softmax special case, closed form
                                     (the standing IIA comparison and control
                                     variate; exact inverse abilities_from_softmax)
    race_jacobian(mu, ...)           fixed-grid dp/dmu      (factor.polish)
    polish_race(p, ..., caps)        nearest race obeying linear constraints
    top_k_probabilities(mu, k, ...)  P(among the k best), min-wins
    bottom_k_probabilities(mu, k)    P(among the k worst)
    removal_shares(mu, ...)          P(j wins | i removed), all pairs, one field
    tie_densities(mu, ...)           photo-finish weights (circuit conductances)

Block/nested/tree kernels: winning.factor.blocks (fastrace-accelerated where
available); inversion for blocks via abilities_from_block_race.

winning.probit speaks the factor-probit literature's language (max-wins
utilities, shares, supplied covariances). winning.factor holds the
paper-faithful kernels; winning.thurstone the density-agnostic engine;
winning.classic the original SIAM lattice ability transform, whose
primitive is the OPPOSITE of the front door's: there the atom vector on
the lattice IS the distribution (empirical data, real dead heats,
multiplicity calculus), here a distribution is a formula evaluated
exactly with only the win integral discretized. Route by provenance:
formulas here, atoms there. (The old top-level imports alias to
winning.classic with a DeprecationWarning.)
calibrate_factors (outer estimation) is reserved for a future release.
"""

from .factor import (  # noqa: F401
    calibrate_abilities,
    race_probabilities,
    removal_shares,
    tie_densities,
)
from .factor.races import (  # noqa: F401
    abilities_from_softmax,
    harville_order_logprob,
    harville_place_probabilities,
    softmax_probabilities,
)
from .factor.structures import (  # noqa: F401
    Blocks,
    Factor,
    Independent,
    Nested,
    Tree,
)
from .factor.polish import polish_race, race_jacobian  # noqa: F401
from .factor.topk import (top_k_probabilities,  # noqa: F401
                          bottom_k_probabilities)
from . import probit  # noqa: F401
from .rustconfig import use_rust, rust_active  # noqa: F401

__version__ = "1.4.0"
