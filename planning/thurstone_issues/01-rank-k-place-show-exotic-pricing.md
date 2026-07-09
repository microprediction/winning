# Rank-k, place/show and exotic pricing (Harville territory, done right)

**Type:** enhancement

thurstone currently prices only the winner (`Race.state_prices`, `winner_of_many`).
winning 1.x also priced finishing positions and combinations, which is where the
lattice model most visibly beats the Harville formula (see the SIAM paper's
comparisons against Harville and the bookmakers' "rule of a quarter", reproduced in
winning's `papers/siam2021/`):

- `five_prices_from_five_densities` — probability of finishing 1st..5th for each
  contestant (this was the M6 competition primitive).
- `skew_normal_place_pricing` — win odds in, place/show prices out (Monte Carlo after
  calibration).
- `placegetters_from_performances`, `exotic_count` — exacta/trifecta counting.

Proposal: rank-k probabilities belong in thurstone core next to `winner_of_many` —
the "winner of the rest, after removing the winner" recursion (Plackett peeling on
the lattice) gives place/show analytically, no Monte Carlo needed for small k.
Exotics (exacta/trifecta) could stay downstream in `winning`, which now handles
rating/racing applications.

Reference implementation to port from: `attic/lattice_simulation.py` in the winning
repo (git history: `winning/lattice_simulation.py` at v1.0.6).
