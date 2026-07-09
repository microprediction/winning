# The rating laboratory: market-implied truth + the inspiration simplex

*Vision note, July 2026. Follows the market-hybrid and beat-the-market
experiments in research/.*

## The problem with judging rating systems by outcomes

An outcome is one draw from the probabilities we care about. Scoring a system
by log loss on results needs thousands of events to resolve small quality
differences, and the noise floor rises with field size. This is why the
synthetic worlds carry an oracle: `Event.truth` lets `evaluate` report
`tv_vs_oracle`, comparing full probability vectors per event instead of one
Bernoulli draw — an order of magnitude more statistical power.

## Market-implied truth makes a real-data oracle

Where a liquid market exists, its recalibrated probabilities are the best
available estimate of the truth. Plan:

1. Fit the market temperature `a` (favorite-longshot correction) on HK racing
   — the beat-the-market experiment produces exactly this, leakage-free.
2. Attach `truth_i ∝ market_i^a` to every HK event (a loader option, e.g.
   `hkracing_events(oracle=True)`); optionally also invert through
   `thurstone.AbilityCalibrator` to get per-race "true relative abilities".
3. Every purely fundamental (outcome-only) rating system is then judged by
   `tv_vs_oracle` on real races: a stern, high-power test that single-draw
   outcomes cannot provide. Judge by probabilities, by abilities, or both.

The dataset stops being a contest the market always wins and becomes an
instrument for measuring how much of the market's knowledge a fundamental
system can reconstruct from outcomes alone.

## Why the lab matters: transfer to marketless domains

Systems developed against market-implied truth deploy where no market exists.
The sharpest target is LLM evaluation: Arena-style leaderboards are contests
— pairwise/listwise preferences, heavy ties, abilities that drift as models
update — with no prices anywhere. Every mechanism in this package (dead-heat
groups, time diffusion, exact order statistics, arbitrary performance
densities) applies verbatim; thurstone's BACKGROUND.md already draws the
DPO/Bradley-Terry connection. A rater that provably tracks market truth on
HK racing is a rater you can trust to rank models.

## The inspiration simplex over rating systems

Per "The Inspiration Simplex" (Cotton 2026, humpday repo: the construction
that generated a derivative-free optimizer beating its five ancestors out of
sample): place established methods at simplex vertices, bind barycentric
weights to functional slots, let a language model realize each coordinate as
a working program, and search the simplex with standard DFO.

Rating systems fit the construction unusually well:

- **Vertices**: Elo, Glicko-2, TrueSkill, OpenSkill-PlackettLuce,
  ThurstoneRating — all already implemented here behind one interface
  (`observe/win_probabilities/rating/elapse`), the fixed calling convention
  the map needs.
- **Slots**: belief representation (point / Gaussian / lattice density);
  update rule (pairwise delta / EP moment-match / exact stage likelihood);
  prediction formula (BT softmax / pairwise probit / exact winner-of-many);
  time dynamics (none / RD inflation / diffusion); tie-and-field handling.
- **Objective**: the lab. TV-to-oracle on HK racing as the primary (smooth,
  high-power) score, with the twelve-dataset suite as held-out validation —
  the out-of-sample discipline of the paper carries over directly.
- **Outer loop**: humpday optimizers on the simplex.

thurstone underneath, winning as the lab, humpday driving the search: one
experiment across all three packages.

## Concrete next steps

1. Land the fitted temperature from research/beat_the_market.py; add the
   oracle option to the HK loader and a `hkracing-lab` dataset variant.
2. Publish the first "fundamental systems vs market truth" table
   (tv_vs_oracle for all systems) in BENCHMARKS.md.
3. Ability-space scoring: invert oracle probs per race, compare to each
   system's pre-race rating vector (rank correlation / RMSE on overlap).
4. An LLM-preference dataset (e.g. Chatbot Arena public data) as the first
   marketless deployment of whatever the lab certifies.
5. The simplex experiment proper, once 1-4 are stable.
