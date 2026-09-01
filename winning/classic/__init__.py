"""winning.classic: the original lattice ability transform (winning <= 1.x).

The density-agnostic lattice API from the SIAM J. Financial Mathematics
paper: densities on a unit lattice, winner-of-many by iterated
convolution, dead heats, state prices, and the ability transform, for
any base distribution. Racing and betting-market vocabulary
(dividends, state prices) lives here.

This is maintained, parity-locked (the R and JavaScript ports replay
the same embedded vectors), and rust-accelerated -- classic, not
abandoned. The choice between this engine and the front door
(race_probabilities / abilities_from_race) is a PROVENANCE rule, not a
new-versus-old rule. The two engines take different primitives:

  * here, the atom vector IS the distribution -- any probabilities on
    a lattice, from anywhere (empirical finish-time histograms,
    integer scores, bootstrap output). Atoms tie with real positive
    probability, and the multiplicity calculus prices those dead heats
    exactly. Exact for discrete laws; representing a CONTINUOUS law
    here means sampling it onto atoms first, an O(unit^2) conversion
    (measured ~1e-3 at 129 points) the front door does not pay.
  * the front door's distribution is a FORMULA (a standardized
    survival/density callable) evaluated exactly at quadrature points;
    only the win integral is discretized, spectrally. It cannot ingest
    atoms without smoothing them -- which erases genuine dead-heat
    mass no refinement recovers.

So: distributions that arrive as data on a grid belong here;
distributions that arrive as formulas belong at the front door. That
primitive change, not namespace tidiness, is the real reason the API
moved.

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
