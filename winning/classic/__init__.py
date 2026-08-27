"""winning.classic: the original lattice ability transform (winning <= 1.x).

The density-agnostic lattice API from the SIAM J. Financial Mathematics
paper: densities on a unit lattice, winner-of-many by iterated
convolution, dead heats, state prices, and the ability transform, for
any base distribution. Racing and betting-market vocabulary
(dividends, state prices) lives here.

This is maintained, parity-locked (the R and JavaScript ports replay
the same embedded vectors), and rust-accelerated -- classic, not
abandoned. New work should normally use the front door instead:
race_probabilities / abilities_from_race in the top-level package,
which cover the independent case as the trivial grammar.

The old top-level import paths (winning.lattice_calibration, ...) keep
working as deprecation shims that re-export from here.
"""

from winning.classic.lattice_calibration import (  # noqa: F401
    ability_implied_dividends,
    dividend_implied_ability,
    normalize_dividends,
    state_price_implied_ability,
)
from winning.classic.lattice import (  # noqa: F401
    densities_from_offsets,
    skew_normal_density,
    state_prices_from_offsets,
)
