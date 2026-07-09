# Longshot-bias adjustment for market dividends

**Type:** enhancement (small)

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
