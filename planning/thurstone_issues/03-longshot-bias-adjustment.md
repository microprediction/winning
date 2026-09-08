# Longshot-bias adjustment for market dividends

**Type:** enhancement (small)

**STATUS (2026-09-08): NOT FOR PUBLICATION.** Peter's standing rule
is that we publish nothing about horseracing, and this draft is
entirely about betting markets: bookmaker and parimutuel prices, the
favourite-longshot bias, dividend adjustment. The underlying idea (a
power-law reweighting before inversion) is not itself racing and
could be raised in neutral terms if it is ever wanted, but this text
does not get posted anywhere.

winning 1.x shipped `longshot_adjusted_dividends` — a power-law correction applied to
market dividends before inversion, acknowledging the favourite-longshot bias in
betting markets. thurstone's `AbilityCalibrator.solve_from_dividends` takes dividends
at face value.

Proposal: an optional `longshot_exponent` (or a small `MarketAdjustment` hook) on the
dividend path, defaulting to no-op. It matters for anyone feeding real bookmaker or
parimutuel prices rather than exchange prices; the calibration literature typically
uses p_adj ∝ p^lambda with lambda slightly above 1.

Reference implementation: `longshot_adjusted_dividends` in `attic/lattice_simulation.py`
(winning repo).
