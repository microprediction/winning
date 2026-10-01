# Hybrid A: GHK sampling in factor space, the lattice for the conditional race

An experiment, not a feature. `hybrid_a.py` is self-contained: `winning`,
numpy, scipy. Run it on any machine with `winning` installed.

## The question

Winning's engine and GHK are the same move on different variables. Both
condition, then integrate what is left exactly:

- **The lattice** conditions on the shared factors `f` (r dimensions, Q fixed
  nodes). Given `f` the runners are independent and the whole N-runner race is
  one 1-D integral over the shared survival product: exact, O(N L), every
  winner at once. Its only error is the quadrature over `f`, and that fails
  when the conditional race is sharp in `f` -- the integrand is a near-step
  and fixed nodes land on its flat parts.
- **GHK** conditions sequentially on the runners' contrasts (N-1 dimensions,
  R draws). Each draw walks the Cholesky, samples each contrast from its
  truncated conditional, and the weight is a product of 1-D normal CDFs. No
  draw is wasted and no structure is assumed. Cost O(N^2) per draw; stochastic.

Hybrid A keeps the lattice for the conditional race and replaces the fixed
`f` nodes with GHK's recursion **over the r factor dimensions**: for winner
`i`, draw `f_1` from N(0,1) truncated to where `i` is still competitive
given `f_1`, then `f_2` given `f_1`, and so on; weight by the truncation
masses; price `p_i(f)` exactly with the lattice at each draw. "Competitive"
is a polytope in `f` (one linear constraint per opponent, at a margin of a
few idiosyncratic sd), and its exact projection onto each coordinate is two
small LPs -- that is what makes this GHK rather than a box heuristic.

**Hypothesis:** in the sharp, low-to-moderate-rank regime where fixed nodes
fail, draws that go where `p_i(f)` varies beat nodes that sit where it is
flat, at equal lattice passes.

## What is already known (2026-09-22, one machine)

**Rank 1: the hybrid loses.** On an 8-runner field at sharpness 21-47,
scrambled Sobol at 128 nodes reaches 6.5e-4 / 9.9e-4 for all winners at once;
the per-winner hybrid needs ~1400 passes to match it. A single truncation
removes little at rank 1, per-winner draws multiply the passes by n, and the
truncated-normal proposal does not follow the bump of `p_i(f)` (R=50 came out
worse than R=16, raw sum 0.992: mass leaking past the margin). GHK's power in
N-1 dimensions comes from the product of truncations shrinking the region;
one dimension gives one weak cut.

**Found instead:** the shipped rank-1 rule handed over from Gauss-Hermite to
its midpoint-quantile grid too late (Q > 201, sharpness ~25); GH was already
2.8-4.0e-3 off at sharpness 15-21 while the grid sat at the truth floor. Fixed
in winning PR #152 (threshold now Q > 80).

**Open:** whether the sequential truncations compound favourably at rank 3-5,
where GHK's strength actually lives and where the shipped Sobol rule is at
its weakest (measured 1.2e-3 at rank 2 / 2048 nodes on the smoke field). That
is what this script is for.

## How to run

Thread caps are set in-process (3 threads). One job at a time; the LP-per-
draw-per-dimension makes the hybrid slow at rank 4-5 with large R.

```bash
# rank 1 sanity (should reproduce the "hybrid loses" result, ~1 min)
python research/factor_ghk/hybrid_a.py --rank 1 --n 8 --D 0.05 0.01 --paths 3000000

# the open question: rank 3, two sharpness levels, decent budgets (~10-20 min)
python research/factor_ghk/hybrid_a.py --rank 3 --n 8 --D 0.1 0.05 --R 32 128 512 --paths 3000000 --csv r3.csv

# rank 5 (slow: R=128 at rank 5 is ~5 LPs x 128 draws x 8 winners; budget accordingly).
# The fixed Gauss-Hermite baselines use orders chosen by rank (Q = 7, 11 here) so the
# pruned tensor stays under ~3e5 nodes; Q=41 at rank 5 was 116M nodes before pruning (#155).
python research/factor_ghk/hybrid_a.py --rank 5 --n 8 --D 0.1 --R 32 128 --paths 3000000 --csv r5.csv
```

Knobs: `--D` sets sharpness (smaller D, sharper; the printout reports
`sharpness = sqrt(2) max_i ||V_i - mean|| / sqrt(D_i)`); `--margin` is the
competitive margin in idiosyncratic sd (default 4; try 3 and 6); `--points`
the lattice resolution (257 default; 501 to rule the lattice out as the
error source); `--paths` the Monte Carlo truth (3M gives se ~3e-4; do not
read differences below ~2 se).

## What to look for

- **Success** = Hybrid A (normalised) error at or below scrambled Sobol's at
  equal `passes`. Compare the per-winner hybrid at `R` against Sobol at
  `~n*R` nodes, and the shared hybrid at `R` against Sobol at `~R`.
- **Raw sum far from 1** means the margin is cutting off mass (too small) or
  the proposal is missing the bump (variance). Report both raw and normalised;
  if normalising is doing most of the work, the estimator is not yet right.
- **Per-winner vs shared.** Per-winner is the faithful GHK analogue but costs
  n passes per draw. Shared draws from a mixture cost one pass per draw but
  need the mixture density, which recomputes every winner's chain mass at each
  `f` (the expensive inner loop). If shared wins at equal passes, the cost
  objection to the hybrid disappears.
- **Sharpness dependence.** The hybrid's case is the sharp regime. If it only
  wins at sharpness > 40 where the shipped rule already escalates, it is not
  worth shipping.

Record results next to this file (`RESULTS.md`, with the machine, the command,
and the table). If it wins at rank 3-5: the follow-up is a node family in
`races._setup`'s escalation, chosen when `r >= 2` and the sharpness statistic
crosses the measured threshold, with the midpoint grid staying for rank 1.
If it does not: the shipped rule plus #152 stand, and `qmc_ghk` remains the
dense-covariance path.
