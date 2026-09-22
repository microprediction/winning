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
flat, at equal lattice passes. **Outcome at n=8: rejected at every rank 1-5.
Large n: `RESULTS.md` part 2.**

## Result (2026-09-22), part 1: at n=8 the hybrid loses at every rank

See `RESULTS.md` for the tables, the machine, and the certification of
the truth. In short: at n=8, ranks 1 to 5, sharpness 13 to 47, margins 3,
4 and 6, Hybrid A never reaches scrambled Sobol's error at equal lattice
passes, and the shared variant is worse than per-winner. The raw sums sit
within 0.5% of 1, so the truncation removes almost nothing: r wide cuts
in factor space are not N-1 sharp cuts on the contrasts, and that was the
whole mechanism at n=8.

## Result (2026-09-22), part 2: at n = 1,000 to 100,000 it matches per pass, loses on the clock

The regime winning exists for. `hybrid_a_largen.py` prices 12 target
runners (8 favourites, 4 mid-field down to p = 3e-4) with the per-winner
hybrid and scores fixed Sobol nodes on the same targets, against a 2^17
to 2^18 Sobol truth computed in parallel and certified against a second
scramble. Here the truncation does remove mass (half the prior for the
favourite at n=100,000 from the first cut alone) and the hybrid goes from
losing outright to roughly matching Sobol per pass on a dozen targets,
with a 4 to 8x per-target advantage in passes for a single runner and a 1
to 2x edge in relative error on longshots. It still loses on wall time,
because every draw costs two LPs per factor dimension with n-1
constraints, and it cannot price the field, which is the job. GHK on the
runner contrasts is measured dead as a field method by n=10,000 (32 s per
target at 256 draws, 88 hours for all winners; 80 GB covariance at
n=100,000). Tables and the verdict in `RESULTS.md` part 2.

What the run found instead: at n=8 the shipped `qmc_ghk` beats the
lattice's default rule on accuracy and time once the rank reaches 4
(details in `RESULTS.md`). The rank 1 finding from the first pass (the
Gauss-Hermite handover was too late) went into winning PR #152.

The first pass's rank 1 figure ("the per-winner hybrid needs ~1400 passes
to match Sobol at 128") came from a 3M-path Monte Carlo truth with a
standard error of 3e-4 and should be discarded; against the certified
truth the gap at rank 1 is 1.5 to 4x in passes, and the rank 1 field on
this seed is a two-horse race, so the rank 1 rows say little either way.

Two defects in the first pass were fixed and matter for anyone reusing
the script. The 3M-path MC truth (se ~3e-4) could not resolve the
question; the truth is now a 2^18 to 2^20 scrambled-Sobol rule with its
own seed, certified against a second scramble and against MC on every
field. And `winning.methods.native._ghk_prob` is max-wins, so the GHK
baseline must be fed `-mu`.

## Result (2026-09-22), part 3: shared node rules, composites, screening

Follow-ups tested against the same truths (`nodes_b.py`, `composite.py`,
`screen.py`). Rotating the factors so Sobol's first coordinate is the
sharp direction: no effect (a reversed control matched it). An adaptive
stratified tree that splits where p(f) varies: 2 to 10x worse than plain
Sobol. Composite horses (one runner standing in for a group of poor ones,
with the exact group-minimum distribution and mean loading): fails
badly, because a poor runner's relevance is factor geometry and the
within-group loading spread is 7x the idiosyncratic sd. Pruning from a
pilot: exact at margin 5 but keeps 15 to 40% of the field. **Per-node
contender screening**: at each node drop the runners with no mass below
the winner distribution's upper 1e-12 quantile, then price the rest;
exact to 1e-10 against the engine and 10x faster at n=10,000, 25x at
n=100,000 (448 ms to 18 ms per node). Neither the numpy pass nor the Rust
kernel (`rust/winning/src/lib.rs`, `forward_kernel`) screens today; both
price all n runners at every node against one shared window. That one is
worth shipping in both, with the "wavefront" tail for the skipped runners
(a two-horse race against the winner distribution the contender lattice
already produces; +7% cost, no zeros, exact to 1e-20, overstates beyond).
Details and tables in `RESULTS.md` part 3.

## How to run

The script is resumable. Running sums for every stochastic method live in
`runs/state/<field>.json`; rerun with a larger `--R`, `--ghk` or
`--truth-m` and only the new draws are priced. The CSV is appended and
flushed one row at a time. `winning` is imported from the checkout, so
set `PYTHONPATH=.` from the repo root. For long runs use `caffeinate -i`.

```bash
# rank 1 (a two-horse race on this seed; kept for completeness, ~1 min)
PYTHONPATH=. python3 research/factor_ghk/hybrid_a.py --rank 1 --n 8 --D 0.05 0.01 \
    --R 32 128 512 2048 --ghk 128 512 2048 8192 32768 --paths 3000000 --csv research/factor_ghk/runs/sweep_rank1.csv

# ranks 2-5, three sharpness levels each, plus margins 3 and 6 at rank 3: 14 processes, ~7 min on 24 cores
research/factor_ghk/runs/sweep.sh

# more accuracy later: same command with larger budgets resumes from the state files
PYTHONPATH=. python3 research/factor_ghk/hybrid_a.py --rank 3 --D 0.05 --R 32 128 512 2048 8192 --ghk 128 512 2048 8192 32768 131072 \
    --truth-m 20 --csv research/factor_ghk/runs/sweep_rank3_D0.05.csv

# rebuild the n=8 tables in RESULTS.md
python3 research/factor_ghk/compile_results.py

# large n: 12 targets, parallel truth, resumable the same way (state in runs/state_largen/)
research/factor_ghk/runs/sweep_largen.sh            # n = 1e3, 1e4, 1e5; ~1.5 h on 24 cores
research/factor_ghk/runs/resume_n100000.sh          # n = 1e5 resumed to a 2^17 truth and R=2048; ~2 h
PYTHONPATH=. python3 research/factor_ghk/hybrid_a_largen.py --n 1000 --rank 3 --D 0.05 --R 32 128 512 2048 \
    --sobol-m 7 9 11 13 15 --truth-m 18 --ghk 256 1024 4096 --workers 24 --csv research/factor_ghk/runs/largen_n1000.csv
python3 research/factor_ghk/compile_largen.py

# part 3: shared node rules (rotation, adaptive tree; pruning at large n), composites, screening
research/factor_ghk/runs/nodes_b.sh
PYTHONPATH=. python3 research/factor_ghk/composite.py --n 1000 --m 9 11 --groups 0 5 20 --validate
PYTHONPATH=. python3 research/factor_ghk/screen.py --n 100000 --m 13 15     # screened_pass(..., tail=True) adds the wavefront tail
```

The n=100,000 state arrays (6.4 MB, eight vectors of 100,000 doubles)
are not committed; a resume at that n recomputes the 2^17 truth, about
70 minutes on 24 workers. The other state files are.

Large-n knobs: `--top` and `--mid` choose the targets; `--ghk-max-n` and
`--ghk-targets` bound the GHK spot-check (each worker holds about three
n^2 doubles); `--workers` is the pool size, one BLAS thread each. Truth
streams extend in place, so a larger `--truth-m` prices only the new
nodes; a smaller one than the state holds is reported at the stored size.

Knobs: `--D` sets sharpness (smaller D, sharper; the printout reports
`sharpness = sqrt(2) max_i ||V_i - mean|| / sqrt(D_i)`); `--margin` is the
competitive margin in idiosyncratic sd (default 4); `--points` the lattice
resolution (257; the conditional race is priced to ~1e-15 at any value from
129 up, measured); `--truth-m` the truth's Sobol exponent (18 at rank 2,
20 at rank 3 and above; the printout reports the truth's disagreement with
a second scramble, and nothing below twice that is readable); `--paths`
the MC paths used only to certify the truth. Budgets must be ascending and
at least what the state file already holds.

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

Results are in `RESULTS.md` (machine, commands, tables). It did not win,
so the shipped rule plus #152 stand and `qmc_ghk` remains the
dense-covariance path. The one follow-up the data suggests is on the
baseline, not the hybrid: the rank >= 4 default rule at small n is beaten
by `qmc_ghk` on both accuracy and time, so a GHK handover or more Sobol
nodes there is worth measuring.
