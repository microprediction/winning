"""exp29c: why the chess factor rating collapsed — the two diagnostics.

D1 (self-selection): per-player tactical share as White is bimodal at
the extremes — deciles 0.00/0.10/0.86/0.98/1.00, sd 0.415 against a
no-choice binomial null of 0.058 (7x). Players do not sample openings;
they ARE their openings. The covariate is nearly a deterministic
function of the player, so per-player offsets are unidentified for
most of the population and selection compresses differential skill in
the realized games.

D2 (the axis exists anyway): among the 790 players with >=20 games in
EACH family, the cross-player variance of (tactical - positional win
rate) exceeds its binomial noise floor by +0.00222 — a true style sd
of about 4.7 win-rate points, roughly 4 sigma above noise. (Caveat:
opponent-strength composition differs across a player's two diets, so
this is a diagnostic, not a clean estimate.)

Together with exp29b (fair fit collapses onto the scalar, offset sd
0.0078, difference -0.0000 [-0.0001,+0.0000]): the positional/tactical
axis is real and observational game data cannot identify it, because
choice destroys within-player covariate variation. Identification
would need exogenous assignment (thematic events) or the mixed-diet
subpopulation.
"""
print(__doc__)
