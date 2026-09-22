# Hybrid A results (2026-09-22)

Two parts. Part 1 is the README's original sweep at n=8. Part 2 is the
regime winning exists for, n = 1,000 to 100,000, where the competitive
polytope is small and the truncation has something to remove.

# Part 1: n=8, ranks 1 to 5

Machine: Apple M3 Ultra, 96 GB, macOS 26.6.2, Python 3.14, numpy 2.4.3,
scipy 1.17.1, `winning` at b9b5645, numpy lattice path (no Rust). Run
via `runs/sweep.sh` (14 fields in parallel, 2 BLAS threads each, ~7 min
wall) plus the rank 1 command in the README. Every row lives in
`runs/sweep_*.csv`; the tables below are `python compile_results.py`.

## Verdict

**At n=8, Hybrid A loses at every rank from 1 to 5, at every sharpness
tried, and at every margin.** (Large n is Part 2; the verdict there is
separate.) It never reaches scrambled Sobol's error at equal
lattice passes, and it is not close: at rank 3, per-winner Hybrid A with
16k passes sits at 3e-4 to 6e-4 while Sobol with 8k passes sits at 5e-5
to 2e-4. At rank 4 and 5 the two are within a factor of two of each other,
but the hybrid uses twice the passes and 150x the wall time. The shared
variant is worse than per-winner at equal passes everywhere. Margins 3
and 6 at rank 3 move the answer by less than the run-to-run noise.

Why it fails at n=8, as far as the numbers say:

- The truncation removes little. Raw sums sit within 0.5% of 1 at
  R=2048 (last table), so the competitive polytope at margin 4 contains
  nearly all the mass. The proposal is then close to the prior, and the
  weighting buys nothing over plain nodes.
- The proposal does not follow `p_i(f)`. GHK's variance reduction comes
  from N-1 sequential cuts that each remove most of the remaining region.
  In factor space there are r cuts and each is wide.
- Per-winner draws multiply the passes by n for the same draw count, and
  the shared mixture pays that back in importance-weight variance.

## What was fixed on the way

- **The truth.** The committed script scored against 3M-path Monte Carlo,
  standard error about 3e-4, which cannot separate methods at the 1e-4
  level the question needs. The lattice prices the conditional race to
  ~1e-15 at any resolution (points 129 to 2001 agree to 1e-14), so the
  only error is the quadrature over `f`, and a 2^18 to 2^20 scrambled
  Sobol rule with a seed the compared rules never use is a far better
  reference. It is certified against a second independent scramble and
  against MC on every field (first table). Read nothing below twice the
  scramble disagreement.
- **The GHK baseline was max-wins.** `winning.methods.native._ghk_prob`
  computes P(i has the max); winning is min-wins. Fed `mu` directly it
  scores 0.57 against the truth. The script now passes `-mu`.
- **Batched lattice pricing.** The engine normalises its output to sum 1
  whatever the weights, and each conditional race sums to 1 exactly, so
  the un-normalised importance sum is `p_norm * sum(w)`. One engine call
  per winner replaces one per draw; results agree with the per-draw loop
  to 1e-16. The LP solves, not the lattice, now dominate the hybrid's
  wall time.
- **Resumable.** Every stochastic method is written as a chunk over draw
  indices of a fixed scrambled-Sobol stream, with running sums in
  `runs/state/*.json`. Rerunning with a larger `--R`, `--ghk` or
  `--truth-m` prices only the new draws. The CSV is appended and flushed
  per row.

## A finding about the baseline at small n, not the hybrid

At n=8 the shipped `qmc_ghk` beats the lattice's default rule on both
axes once the rank reaches 4. GHK at 32768 draws sits at 1e-5 to 5e-5 in
0.05 s on every field, at the truth floor for rank 5. The default rule is
8192 Sobol nodes in 0.3 s, at 4e-5 for rank 2 but 9e-4 to 1.3e-3 at rank
4 and 3e-4 to 6e-4 at rank 5. GHK at 2048 draws already beats the default
rule at rank 4 and 5 in a hundredth of the time. This is the small-n
regime; the lattice's case is large N, where GHK's per-winner O(N^2)
becomes O(N^3) for all winners (experiment 13 has the frontier). It does
say the rank >= 4 default rule at small n has room, and the fix there is
more Sobol nodes or a GHK handover, not Hybrid A.

Rank 1 is a two-horse race on this seed (p = 0.525, 0.475, rest below
3e-5), so the rank 1 rows say little; they are kept because the README
promised them.

### Fields and truth certification

| field | sharpness | truth nodes | truth vs 2nd scramble | truth vs MC 3M (se ~3e-4) | hybrid passes at R=2048 |
|---|---:|---:|---:|---:|---:|
| r1 D0.05 | 21.1 | 2^18 | 1.7e-08 | 2.6e-05 | 14336 |
| r1 D0.01 | 47.1 | 2^18 | 1.8e-11 | 6.4e-05 | 6144 |
| r2 D0.1 | 14.0 | 2^18 | 4.6e-07 | 2.1e-04 | 16384 |
| r2 D0.05 | 19.8 | 2^18 | 8.3e-07 | 3.4e-04 | 16384 |
| r2 D0.02 | 31.3 | 2^18 | 1.3e-06 | 4.2e-04 | 14336 |
| r3 D0.1 | 13.3 | 2^20 | 3.2e-06 | 2.3e-04 | 16384 |
| r3 D0.05 m3 | 18.8 | 2^20 | 3.8e-06 | 1.6e-04 | 16382 |
| r3 D0.05 | 18.8 | 2^20 | 3.8e-06 | 1.6e-04 | 16383 |
| r3 D0.05 m6 | 18.8 | 2^20 | 3.8e-06 | 1.6e-04 | 16384 |
| r3 D0.02 | 29.8 | 2^20 | 5.4e-06 | 3.1e-04 | 16382 |
| r4 D0.1 | 13.3 | 2^20 | 8.1e-06 | 3.1e-04 | 16383 |
| r4 D0.05 | 18.8 | 2^20 | 1.6e-05 | 3.4e-04 | 16383 |
| r4 D0.02 | 29.7 | 2^20 | 2.6e-05 | 1.8e-04 | 16383 |
| r5 D0.1 | 15.4 | 2^20 | 1.1e-05 | 4.1e-04 | 16347 |
| r5 D0.05 | 21.7 | 2^20 | 1.5e-05 | 5.0e-04 | 16344 |
| r5 D0.02 | 34.3 | 2^20 | 4.3e-05 | 5.5e-04 | 16343 |

### max |p - truth| over all winners, normalised estimates

| method | r1 D0.05 | r1 D0.01 | r2 D0.1 | r2 D0.05 | r2 D0.02 | r3 D0.1 | r3 D0.05 m3 | r3 D0.05 | r3 D0.05 m6 | r3 D0.02 | r4 D0.1 | r4 D0.05 | r4 D0.02 | r5 D0.1 | r5 D0.05 | r5 D0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default rule (= Sobol 2^13 at rank >= 2) | 3.1e-06 | 5.2e-10 | 3.8e-05 | 3.8e-05 | 4.3e-05 | 5.5e-05 | 9.5e-05 | 9.5e-05 | 9.5e-05 | 2.4e-04 | 8.7e-04 | 1.0e-03 | 1.3e-03 | 3.2e-04 | 4.2e-04 | 6.1e-04 |
| Sobol 2^7 (128 passes) | 2.7e-04 | 6.1e-04 | 4.8e-03 | 4.3e-03 | 3.3e-03 | 1.2e-02 | 1.4e-02 | 1.4e-02 | 1.4e-02 | 1.7e-02 | 1.9e-02 | 2.1e-02 | 2.1e-02 | 7.5e-03 | 9.5e-03 | 1.2e-02 |
| Sobol 2^9 (512) | 3.6e-06 | 8.7e-06 | 7.1e-04 | 5.2e-04 | 5.2e-04 | 3.4e-03 | 4.5e-03 | 4.5e-03 | 4.5e-03 | 5.8e-03 | 8.1e-03 | 9.3e-03 | 9.8e-03 | 1.0e-02 | 1.1e-02 | 1.1e-02 |
| Sobol 2^11 (2048) | 7.1e-07 | 2.9e-07 | 2.6e-04 | 4.2e-04 | 6.7e-04 | 4.8e-04 | 8.5e-04 | 8.5e-04 | 8.5e-04 | 1.4e-03 | 1.9e-03 | 2.5e-03 | 3.5e-03 | 1.3e-03 | 2.1e-03 | 2.7e-03 |
| Sobol 2^13 (8192) | 1.9e-06 | 7.7e-10 | 3.8e-05 | 3.8e-05 | 4.3e-05 | 5.5e-05 | 9.5e-05 | 9.5e-05 | 9.5e-05 | 2.4e-04 | 8.7e-04 | 1.0e-03 | 1.3e-03 | 3.2e-04 | 4.2e-04 | 6.1e-04 |
| Hybrid A per-winner R=32 (~256 passes) | 2.2e-04 | 6.8e-04 | 8.7e-03 | 1.1e-02 | 9.1e-03 | 2.2e-02 | 1.3e-02 | 2.0e-02 | 2.5e-02 | 1.2e-02 | 2.7e-02 | 1.5e-02 | 1.3e-02 | 1.5e-02 | 1.8e-02 | 1.7e-02 |
| Hybrid A per-winner R=128 (~1024) | 1.4e-05 | 1.4e-05 | 1.4e-03 | 2.7e-03 | 3.0e-03 | 3.9e-03 | 4.2e-03 | 5.4e-03 | 4.0e-03 | 5.6e-03 | 4.6e-03 | 6.3e-03 | 6.5e-03 | 8.0e-03 | 8.7e-03 | 6.4e-03 |
| Hybrid A per-winner R=512 (~4096) | 8.7e-06 | 2.0e-05 | 4.3e-04 | 1.8e-04 | 5.3e-04 | 2.2e-03 | 1.3e-03 | 1.7e-03 | 2.5e-03 | 1.5e-03 | 4.7e-03 | 4.2e-03 | 2.4e-03 | 4.1e-03 | 2.6e-03 | 4.4e-03 |
| Hybrid A per-winner R=2048 (~16384) | 5.5e-06 | 4.0e-08 | 1.5e-04 | 1.1e-04 | 5.0e-05 | 2.9e-04 | 4.2e-04 | 4.0e-04 | 2.5e-04 | 6.0e-04 | 1.7e-03 | 1.6e-03 | 1.4e-03 | 1.2e-03 | 1.3e-03 | 1.3e-03 |
| Hybrid A shared R=256 (256 passes) | 2.8e-03 | 4.6e-03 | 9.6e-03 | 4.2e-03 | 1.9e-02 | 1.4e-02 | 8.6e-03 | 3.0e-02 | 1.3e-02 | 1.3e-02 | 3.5e-02 | 5.8e-02 | 4.5e-02 | 3.3e-02 | 4.9e-02 | 4.3e-02 |
| Hybrid A shared R=1024 (1024) | 3.2e-04 | 8.5e-05 | 1.6e-03 | 3.9e-03 | 4.3e-03 | 5.1e-03 | 8.5e-03 | 7.0e-03 | 3.0e-03 | 6.8e-03 | 6.6e-03 | 1.5e-02 | 7.4e-03 | 1.1e-02 | 1.0e-02 | 1.4e-02 |
| Hybrid A shared R=4096 (4096) | 1.2e-04 | 2.6e-04 | 8.6e-04 | 1.0e-03 | 6.9e-04 | 1.1e-03 | 2.2e-03 | 2.3e-03 | 1.3e-03 | 5.7e-03 | 4.8e-03 | 4.5e-03 | 8.9e-03 | 5.1e-03 | 7.2e-03 | 5.5e-03 |
| Hybrid A shared R=16384 (16384) | 1.7e-05 | 1.6e-05 | 1.2e-04 | 4.1e-04 | 3.7e-04 | 5.5e-04 | 9.0e-04 | 1.2e-03 | 1.6e-03 | 1.2e-03 | 3.1e-04 | 6.2e-04 | 1.6e-03 | 2.2e-03 | 3.6e-03 | 5.2e-04 |
| qmc_ghk B=512 (no lattice) | 6.0e-06 | 5.5e-10 | 1.2e-04 | 1.8e-04 | 1.1e-04 | 4.7e-04 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 2.6e-03 | 8.3e-04 | 1.2e-03 | 6.8e-04 | 1.2e-04 | 3.1e-04 | 6.2e-04 |
| qmc_ghk B=2048 | 3.5e-06 | 5.5e-10 | 3.1e-05 | 9.5e-05 | 9.9e-05 | 1.8e-04 | 2.1e-04 | 2.1e-04 | 2.1e-04 | 2.3e-04 | 1.8e-04 | 1.7e-04 | 1.3e-04 | 7.0e-05 | 8.0e-05 | 1.6e-04 |
| qmc_ghk B=8192 | 7.1e-08 | 5.5e-10 | 2.6e-05 | 5.2e-05 | 1.7e-05 | 7.5e-05 | 1.8e-04 | 1.8e-04 | 1.8e-04 | 3.3e-04 | 1.9e-04 | 2.4e-04 | 2.5e-04 | 3.0e-05 | 8.3e-05 | 1.5e-04 |
| qmc_ghk B=32768 | 1.2e-07 | 5.5e-10 | 2.6e-06 | 1.4e-05 | 2.3e-05 | 8.9e-06 | 1.7e-05 | 1.7e-05 | 1.7e-05 | 2.2e-05 | 1.4e-05 | 1.8e-05 | 1.6e-05 | 2.5e-05 | 2.7e-05 | 5.0e-05 |

### Wall time, seconds (Apple M3 Ultra, 2 BLAS threads, numpy lattice path, LPs via HiGHS)

| method | r1 D0.05 | r1 D0.01 | r2 D0.1 | r2 D0.05 | r2 D0.02 | r3 D0.1 | r3 D0.05 m3 | r3 D0.05 | r3 D0.05 m6 | r3 D0.02 | r4 D0.1 | r4 D0.05 | r4 D0.02 | r5 D0.1 | r5 D0.05 | r5 D0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default rule | 0.01 | 0.02 | 0.34 | 0.33 | 0.30 | 0.31 | 0.33 | 0.31 | 0.31 | 0.29 | 0.32 | 0.31 | 0.30 | 0.31 | 0.31 | 0.31 |
| qmc_ghk B=32768 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| Hybrid A per-winner R=2048 | 11.24 | 7.37 | 27.39 | 27.33 | 24.44 | 40.40 | 40.28 | 40.26 | 40.63 | 40.40 | 54.21 | 54.08 | 54.15 | 66.86 | 67.16 | 67.40 |
| Hybrid A shared R=16384 | 51.85 | 13.96 | 167.46 | 142.30 | 92.57 | 205.90 | 166.64 | 179.50 | 213.12 | 165.11 | 252.60 | 224.32 | 200.81 | 298.72 | 269.37 | 242.17 |

### Raw sums of the hybrid estimates (mass leaking past the margin shows here)

| method | r1 D0.05 | r1 D0.01 | r2 D0.1 | r2 D0.05 | r2 D0.02 | r3 D0.1 | r3 D0.05 m3 | r3 D0.05 | r3 D0.05 m6 | r3 D0.02 | r4 D0.1 | r4 D0.05 | r4 D0.02 | r5 D0.1 | r5 D0.05 | r5 D0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-winner R=2048 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 0.9991 | 0.9990 | 0.9990 | 0.9989 | 0.9991 | 0.9992 | 0.9996 | 0.9995 | 0.9992 | 0.9972 | 0.9957 |
| shared R=16384 | 1.0001 | 1.0001 | 0.9991 | 0.9989 | 1.0003 | 0.9991 | 1.0006 | 1.0013 | 1.0033 | 1.0016 | 0.9963 | 1.0018 | 0.9946 | 0.9982 | 0.9984 | 1.0031 |

# Part 2: n = 1,000 to 100,000, rank 3

Same model, D = 0.05, seed 3, margin 4. Script `hybrid_a_largen.py`, run
via `runs/sweep_largen.sh` and `runs/resume_n100000.sh`; rows in
`runs/largen_n*.csv`, tables from `python compile_largen.py`. The truth
is a 2^17 to 2^18 scrambled-Sobol rule (2^15 at n=100,000 in the first
pass, resumed to 2^17), computed in parallel over node chunks and
certified against a second scramble; the certification line in each table
is the floor below which nothing is readable.

At large n the per-winner hybrid cannot price everyone (n passes per
draw), so it prices 12 targets: the 8 favourites and the runners nearest
p = 1e-2, 3e-3, 1e-3, 3e-4. Fixed Sobol nodes price all n in one pass per
node and are scored on the same targets (and on all n, for the record).
The shared/mixture variant needs n chain masses per draw and is out.
"Passes" for the hybrid is the total over the 12 targets; "per target" is
what one runner costs.

## Verdict at large n

**The truncation bites here, and the hybrid goes from losing outright to
roughly matching Sobol per pass on a dozen targets.** At n=8 the
competitive polytope held all the mass. At n=100,000 a single cut on the
first factor already removes half the prior mass for the favourite, and
more for a longshot. The consequences, from the tables:

- **Absolute error on the favourites, equal total passes:** Sobol wins
  or ties. n=10,000: hybrid at 24,576 passes 1.4e-4; Sobol at 8,192
  passes 4.1e-4 and at 32,768 passes 5.3e-5. n=1,000: hybrid 24,576
  passes 1.5e-4; Sobol 32,768 passes 1.5e-4. n=100,000: hybrid 24,576
  passes 8.4e-5; Sobol 32,768 passes 1.1e-4, both within 2x of the 5e-5
  truth floor, a tie.
- **Per target, the hybrid is 4 to 8x cheaper in passes.** 2,048 hybrid
  passes on one runner reach what Sobol reaches for the field somewhere
  between 8,192 and 32,768 passes. For one or two runners the hybrid is
  the cheaper estimator in passes; for a dozen it is a wash; for the field
  Sobol wins outright, and the field is what winning prices.
- **Relative error on the longshots:** the hybrid is better, but by 1 to
  2x, not the order of magnitude the mechanism promised. n=1,000: hybrid
  R=2048 max relative 7.3e-3 against Sobol 2^15 at 1.3e-2. n=10,000:
  6.9e-3 against 8.2e-3.
- **Wall time is LP-bound and the hybrid loses it.** n=10,000: hybrid
  R=2048 277 s (one process per target) against Sobol 2^15 at 158 s for
  every runner on 12 workers. n=100,000: 2,162 s against 998 s. Each draw costs two LPs per factor dimension
  with n-1 constraints, 30 ms each at n=10,000 and 290 ms at n=100,000.
- **GHK on the runner contrasts is dead as a field method by n=10,000.**
  Per target it is fine at n=1,000 (1.2 s for 4,096 draws) but already
  less accurate than Sobol 2^13 for the whole field in 2.3 s. At n=10,000
  one target at 256 draws costs 32 s for an error of 3.7e-3, which Sobol
  beats in 3 s for all 10,000; all winners would be about 88 hours. At
  n=100,000 the covariance alone is 80 GB.

So Hybrid A is not a replacement for the node rule, which is the job. It
is a defensible way to price one or two runners to 1e-4 with a few
thousand passes when the field does not matter, and even there the LP
cost per draw makes it slower on the clock than pricing everyone with
Sobol. If anything follows, it is a cheaper proposal for the same idea: a
Gaussian fitted to the target's competitive region (one LP per target,
not two per draw per dimension) as an importance density over f, which
would keep the per-target pass advantage and drop the LP-per-draw cost.
That is a different experiment and is not started here.

## Tables

### n = 1,000, rank 3, D = 0.05, sharpness 26.0

Truth: Sobol 2^18 (seed 101), vs second scramble 2.2e-05, vs MC 2.7e-04. Read nothing below 4e-05.

| method | passes | per target | max abs err | max rel err | wall s |
|---|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 128 | 128 | 1.3e-02 | 5.5e-01 | 0.1 |
| fixed scrambled Sobol 2^9 | 512 | 512 | 2.7e-03 | 1.6e-01 | 0.2 |
| fixed scrambled Sobol 2^11 | 2048 | 2048 | 1.5e-03 | 1.2e-01 | 0.6 |
| fixed scrambled Sobol 2^13 | 8192 | 8192 | 2.7e-04 | 5.3e-02 | 2.3 |
| fixed scrambled Sobol 2^15 | 32768 | 32768 | 1.5e-04 | 1.3e-02 | 8.9 |
| Hybrid A per-winner R=32 x 12 targets | 384 | 32 | 1.7e-02 | 3.8e-01 | 0.4 |
| Hybrid A per-winner R=128 x 12 targets | 1536 | 128 | 9.2e-03 | 1.2e-01 | 1.5 |
| Hybrid A per-winner R=512 x 12 targets | 6144 | 512 | 8.7e-04 | 3.0e-02 | 6.0 |
| Hybrid A per-winner R=2048 x 12 targets | 24576 | 2048 | 1.5e-04 | 7.3e-03 | 23.7 |
| qmc_ghk (contrasts) B=256 | - | - | 3.3e-03 | 3.4e-01 | 1.3 |
| qmc_ghk (contrasts) B=1024 | - | - | 2.0e-03 | 1.7e-01 | 3.1 |
| qmc_ghk (contrasts) B=4096 | - | - | 9.6e-04 | 8.3e-02 | 14.6 |

Per-target relative error, favourites left, longshots right:

| method | 1.3e-01 | 1.1e-01 | 6.7e-02 | 6.4e-02 | 4.6e-02 | 3.8e-02 | 3.7e-02 | 3.3e-02 | 9.8e-03 | 2.6e-03 | 9.9e-04 | 3.0e-04 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 4.1e-02 | 1.2e-01 | 7.5e-02 | 1.1e-02 | 2.1e-01 | 1.2e-01 | 1.1e-01 | 4.0e-02 | 1.6e-01 | 2.4e-01 | 3.4e-01 | 5.5e-01 |
| fixed scrambled Sobol 2^9 | 2.0e-02 | 2.2e-02 | 3.2e-02 | 1.9e-03 | 1.7e-02 | 3.3e-02 | 3.6e-02 | 2.3e-02 | 1.9e-02 | 1.6e-01 | 1.3e-01 | 1.0e-01 |
| fixed scrambled Sobol 2^11 | 4.1e-03 | 1.4e-02 | 8.9e-03 | 4.7e-03 | 6.4e-03 | 1.1e-02 | 1.3e-03 | 4.4e-03 | 3.5e-03 | 1.5e-02 | 1.2e-01 | 1.4e-02 |
| fixed scrambled Sobol 2^13 | 2.0e-03 | 1.2e-03 | 3.1e-04 | 6.5e-04 | 4.0e-03 | 6.3e-03 | 2.9e-03 | 8.2e-04 | 6.5e-03 | 3.5e-03 | 1.7e-02 | 5.3e-02 |
| fixed scrambled Sobol 2^15 | 1.2e-04 | 1.4e-03 | 3.0e-04 | 1.8e-04 | 5.1e-05 | 1.6e-03 | 9.8e-04 | 3.0e-04 | 7.2e-04 | 4.4e-04 | 1.4e-03 | 1.3e-02 |
| Hybrid A per-winner R=32 x 12 targets | 1.5e-02 | 1.5e-01 | 7.4e-03 | 4.5e-03 | 2.4e-01 | 2.3e-01 | 1.0e-01 | 3.4e-02 | 1.7e-01 | 1.8e-01 | 3.8e-01 | 3.3e-01 |
| Hybrid A per-winner R=128 x 12 targets | 1.1e-02 | 8.5e-02 | 1.4e-02 | 1.8e-02 | 7.4e-02 | 5.9e-02 | 3.0e-02 | 5.4e-02 | 1.2e-01 | 1.2e-01 | 4.8e-02 | 8.9e-02 |
| Hybrid A per-winner R=512 x 12 targets | 6.4e-03 | 1.2e-03 | 6.3e-03 | 5.6e-03 | 1.7e-03 | 7.1e-03 | 5.1e-03 | 1.7e-03 | 2.5e-02 | 3.0e-02 | 2.1e-02 | 2.3e-02 |
| Hybrid A per-winner R=2048 x 12 targets | 8.4e-04 | 8.0e-04 | 8.0e-05 | 2.4e-03 | 5.2e-04 | 6.8e-04 | 8.5e-04 | 8.5e-05 | 4.4e-03 | 1.7e-03 | 9.6e-04 | 7.3e-03 |
| qmc_ghk (contrasts) B=256 | 9.3e-03 | 1.6e-02 | 2.6e-03 | 9.7e-03 | 2.5e-02 | 8.8e-02 | 7.3e-02 | 1.2e-02 | 9.9e-02 | 3.4e-01 | 2.2e-01 | 1.2e-01 |
| qmc_ghk (contrasts) B=1024 | 1.4e-03 | 1.8e-02 | 1.0e-02 | 1.2e-02 | 6.7e-04 | 2.6e-02 | 4.8e-02 | 1.8e-02 | 7.3e-02 | 1.7e-01 | 1.1e-01 | 2.3e-02 |
| qmc_ghk (contrasts) B=4096 | 5.9e-03 | 7.7e-03 | 7.5e-03 | 2.2e-03 | 1.9e-03 | 3.0e-03 | 2.6e-02 | 1.5e-04 | 7.1e-03 | 5.8e-02 | 8.3e-02 | 1.3e-02 |

### n = 10,000, rank 3, D = 0.05, sharpness 28.5

Truth: Sobol 2^17 (seed 101), vs second scramble 2.7e-05, vs MC 9.3e-04. Read nothing below 5e-05.

| method | passes | per target | max abs err | max rel err | wall s |
|---|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 128 | 128 | 5.4e-03 | 6.9e-01 | 1.1 |
| fixed scrambled Sobol 2^9 | 512 | 512 | 3.5e-03 | 4.5e-01 | 3.2 |
| fixed scrambled Sobol 2^11 | 2048 | 2048 | 1.2e-03 | 7.0e-02 | 11.4 |
| fixed scrambled Sobol 2^13 | 8192 | 8192 | 4.1e-04 | 9.2e-03 | 44.3 |
| Hybrid A per-winner R=32 x 12 targets | 384 | 32 | 1.2e-02 | 3.6e-01 | 4.5 |
| Hybrid A per-winner R=128 x 12 targets | 1536 | 128 | 5.5e-03 | 1.2e-01 | 17.6 |
| Hybrid A per-winner R=512 x 12 targets | 6144 | 512 | 1.1e-03 | 6.2e-02 | 69.8 |
| Hybrid A per-winner R=2048 x 12 targets | 24576 | 2048 | 1.4e-04 | 6.9e-03 | 277.0 |
| qmc_ghk (contrasts) B=256 | - | - | 3.7e-03 | 7.4e-02 | 63.4 |
| fixed scrambled Sobol 2^15 | 32768 | 32768 | 5.3e-05 | 8.2e-03 | 158.5 |

Per-target relative error, favourites left, longshots right:

| method | 6.0e-02 | 4.6e-02 | 4.5e-02 | 4.2e-02 | 4.1e-02 | 4.0e-02 | 3.9e-02 | 3.3e-02 | 9.9e-03 | 3.1e-03 | 1.0e-03 | 3.0e-04 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 8.9e-02 | 1.1e-01 | 2.7e-02 | 6.5e-02 | 7.5e-02 | 5.5e-02 | 1.1e-02 | 2.4e-02 | 1.3e-01 | 2.6e-02 | 1.4e-01 | 6.9e-01 |
| fixed scrambled Sobol 2^9 | 4.9e-03 | 2.0e-02 | 2.7e-02 | 7.2e-02 | 8.4e-02 | 8.4e-03 | 5.5e-02 | 3.6e-02 | 1.1e-01 | 2.1e-01 | 2.8e-01 | 4.5e-01 |
| fixed scrambled Sobol 2^11 | 5.6e-04 | 9.2e-03 | 8.5e-03 | 2.9e-02 | 9.1e-03 | 1.0e-02 | 1.2e-04 | 7.5e-03 | 4.0e-02 | 3.1e-02 | 7.0e-02 | 3.1e-03 |
| fixed scrambled Sobol 2^13 | 1.0e-03 | 3.7e-03 | 9.2e-03 | 8.0e-03 | 4.7e-03 | 2.9e-03 | 4.6e-04 | 1.1e-03 | 2.1e-03 | 6.8e-03 | 7.5e-03 | 1.1e-03 |
| Hybrid A per-winner R=32 x 12 targets | 1.1e-01 | 8.4e-02 | 2.6e-01 | 6.5e-02 | 1.5e-01 | 1.4e-02 | 2.7e-02 | 5.6e-02 | 2.2e-02 | 9.3e-02 | 3.6e-01 | 3.1e-01 |
| Hybrid A per-winner R=128 x 12 targets | 5.2e-02 | 1.9e-02 | 1.2e-01 | 5.6e-02 | 4.7e-03 | 1.0e-02 | 2.0e-02 | 2.6e-02 | 5.7e-02 | 3.3e-02 | 5.8e-02 | 7.7e-02 |
| Hybrid A per-winner R=512 x 12 targets | 1.5e-03 | 3.0e-03 | 7.0e-03 | 7.2e-03 | 3.1e-03 | 6.4e-03 | 2.7e-02 | 3.1e-03 | 1.7e-03 | 8.0e-03 | 3.2e-02 | 6.2e-02 |
| Hybrid A per-winner R=2048 x 12 targets | 1.1e-04 | 3.7e-05 | 2.4e-03 | 9.3e-04 | 3.4e-03 | 4.0e-04 | 1.2e-03 | 3.2e-03 | 4.4e-03 | 6.7e-04 | 6.9e-03 | 3.9e-03 |
| qmc_ghk (contrasts) B=256 | 6.2e-02 | 7.4e-02 | - | - | - | - | - | - | - | - | - | - |
| fixed scrambled Sobol 2^15 | 2.9e-04 | 1.2e-03 | 5.0e-04 | 5.7e-04 | 3.8e-04 | 9.8e-04 | 4.6e-05 | 4.7e-04 | 4.2e-04 | 2.3e-03 | 2.0e-03 | 8.2e-03 |

### n = 100,000, rank 3, D = 0.05, sharpness 31.7

Truth: Sobol 2^15 (seed 101), vs second scramble 1.1e-04, vs MC 1.8e-03. Read nothing below 2e-04.

| method | passes | per target | max abs err | max rel err | wall s |
|---|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 128 | 128 | 5.8e-03 | 7.0e-01 | 8.6 |
| fixed scrambled Sobol 2^9 | 512 | 512 | 2.5e-03 | 3.6e-01 | 22.9 |
| fixed scrambled Sobol 2^11 | 2048 | 2048 | 9.8e-04 | 6.4e-02 | 81.8 |
| fixed scrambled Sobol 2^13 | 8192 | 8192 | 1.7e-04 | 7.5e-03 | 314.8 |
| Hybrid A per-winner R=32 x 12 targets | 384 | 32 | 1.4e-02 | 7.1e-01 | 34.1 |
| Hybrid A per-winner R=128 x 12 targets | 1536 | 128 | 7.9e-03 | 1.4e-01 | 133.7 |
| Hybrid A per-winner R=512 x 12 targets | 6144 | 512 | 1.4e-03 | 4.7e-02 | 529.9 |
| fixed scrambled Sobol 2^15 | 32768 | 32768 | 1.1e-04 | 7.4e-03 | 998.5 |
| Hybrid A per-winner R=2048 x 12 targets | 24576 | 2048 | 8.4e-05 | 1.1e-02 | 2162.4 |

Per-target relative error, favourites left, longshots right:

| method | 1.0e-01 | 5.4e-02 | 4.5e-02 | 3.6e-02 | 3.6e-02 | 3.5e-02 | 3.5e-02 | 3.3e-02 | 1.1e-02 | 3.2e-03 | 1.0e-03 | 2.9e-04 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed scrambled Sobol 2^7 | 1.1e-02 | 7.4e-02 | 9.5e-02 | 7.4e-02 | 3.7e-03 | 5.5e-02 | 1.4e-01 | 5.5e-02 | 5.4e-01 | 7.0e-01 | 2.8e-01 | 4.1e-01 |
| fixed scrambled Sobol 2^9 | 2.3e-02 | 1.7e-02 | 1.6e-04 | 2.2e-02 | 6.2e-03 | 8.1e-03 | 5.1e-02 | 4.6e-02 | 7.8e-02 | 3.4e-02 | 2.3e-02 | 3.6e-01 |
| fixed scrambled Sobol 2^11 | 3.3e-03 | 4.3e-03 | 3.5e-03 | 5.3e-04 | 2.6e-03 | 1.7e-02 | 8.1e-04 | 2.9e-02 | 1.9e-02 | 3.8e-02 | 6.4e-02 | 9.7e-03 |
| fixed scrambled Sobol 2^13 | 7.2e-04 | 2.5e-03 | 1.1e-03 | 4.8e-03 | 1.3e-03 | 4.7e-03 | 3.2e-03 | 4.2e-03 | 4.6e-03 | 2.4e-03 | 7.5e-03 | 4.3e-03 |
| Hybrid A per-winner R=32 x 12 targets | 1.3e-01 | 4.2e-02 | 5.8e-03 | 1.3e-01 | 1.1e-01 | 2.3e-01 | 8.1e-02 | 7.5e-02 | 2.0e-01 | 4.9e-01 | 1.9e-01 | 7.1e-01 |
| Hybrid A per-winner R=128 x 12 targets | 7.5e-02 | 1.3e-02 | 5.5e-02 | 1.4e-02 | 1.1e-02 | 3.3e-02 | 3.3e-02 | 6.9e-03 | 1.8e-02 | 8.7e-02 | 1.4e-01 | 6.8e-03 |
| Hybrid A per-winner R=512 x 12 targets | 3.3e-03 | 2.6e-02 | 3.9e-04 | 7.2e-03 | 3.0e-03 | 1.3e-02 | 6.8e-04 | 6.9e-03 | 1.9e-02 | 2.4e-02 | 1.9e-02 | 4.7e-02 |
| fixed scrambled Sobol 2^15 | 2.8e-04 | 2.0e-03 | 1.2e-03 | 7.6e-04 | 1.2e-03 | 1.5e-03 | 5.4e-04 | 1.9e-04 | 2.6e-03 | 3.1e-03 | 2.0e-03 | 7.4e-03 |
| Hybrid A per-winner R=2048 x 12 targets | 5.6e-04 | 1.1e-03 | 6.8e-04 | 7.9e-04 | 1.0e-03 | 7.2e-04 | 1.5e-03 | 1.5e-03 | 7.8e-03 | 2.7e-03 | 2.6e-03 | 1.1e-02 |


# Part 3: shared node rules, composites, and per-node screening

Ideas raised after Part 2, each tested against the same certified truths
(`nodes_b.py`, `composite.py`, `screen.py`; rows in `runs/nodes_b_*.csv`,
logs in `runs/`). Everything here is shared by all runners: one lattice
pass per node prices everyone, unlike the per-winner hybrid.

## Verdicts

- **Rotation (sharp direction first): no effect.** Aligning Sobol
  coordinate 1 with the top singular direction of the centred loadings,
  plain or weighted by a pilot estimate of who can win, changes nothing
  at ranks 2, 3 and 5 within seed-to-seed noise (4 scrambles per field).
  At rank 4 it looked 2x better on all three fields, but a control that
  puts the sharp direction LAST was equally good, so that is the raw
  coordinates being unlucky on that field, not the direction helping.
  Same at n=1,000. Expected: with 3 to 5 factors every Sobol coordinate
  is already well distributed.
- **Adaptive stratified tree: loses.** Splitting cells of the f-quantile
  cube where the vector p(f) varies most and adding 8 Sobol points per
  child is 2 to 10x worse than plain Sobol at equal passes on every n=8
  field and at n=1,000. Cutting a Sobol set into cells destroys its low
  discrepancy inside them, and the k-point cell means fall back to a
  Monte Carlo rate in the flat regions where most of the volume is.
- **Composite horses: fails, and instructively.** Group the runners who
  never come near the lead, give each group one runner with the group's
  mean loading and the exact distribution of the group minimum with each
  member's loading deviation folded into its own noise. At n=1,000 with
  5 groups the composites "win" 5.5% when their members win 2.5e-10; with
  20 groups, 0.4%. The within-group loading spread is about 1.5 against
  an idiosyncratic sd of 0.22, so a poor runner's relevance is entirely
  factor geometry (it matters only where its factor contrast with the
  field is near its maximum), and no f-independent distribution can
  represent a group of them. Seriation into contiguous 1-D groups is
  worse than k-means. The mini lattice used for this matches the engine
  to 2.5e-14 with no composites, so the failure is the approximation.
- **Pruning from a pilot: exact at margin 5, modest.** Keep the runners
  whose conditional mean comes within `margin` sd of the leader at some
  node of a 2^7 pilot. Margin 5 keeps 39% of the field at n=10,000 and
  15% at n=100,000 and drops 1e-8 of truth mass; the pruned Sobol
  estimates equal the full-field ones. Margin 3 keeps 2 to 9% but drops
  4e-4 of mass at n=100,000, which appears as an error floor at 2.2e-4.
- **Per-node contender screening: exact, 10x at n=10,000, 25x at
  n=100,000.** At each node f, first drop runners whose conditional mean
  minus 8 sd exceeds the lowest upper edge of the field, then find the
  winner distribution's upper 1e-12 quantile (the engine's bulk window,
  but per node, a 50-step bisection over the survivors) and drop
  runners with no mass below it. Price the contenders on that window.
  Matches the engine to 6e-14 (n=1,000), 2e-12 (n=10,000) and 3e-10
  (n=100,000). Contenders per node: 146 of 1,000; 530 of 10,000; 1,630
  of 100,000. Pass cost: 4.6 to 1.1 ms; 45 to 4.6 ms; 448 to 18 ms. Sobol
  2^15 at n=100,000 in 591 s on one process against 998 s for the engine
  on 12 workers. No pilot, no margin, no approximation.

## What this means for winning

Screening is the one item to ship. It is the engine's own bulk window
applied per node with an O(n) skip in front of it, and at n=100,000 it is
the difference between 448 ms and 18 ms per node in the numpy path, or
between a 2^13 rule and a 2^17 rule in the same time. Neither kernel does it
today: `rust/winning/src/lib.rs` `forward_kernel` loops over all n
runners at every node and tile against one window shared by all nodes,
and the numpy path in `races.py` does the same. So the same 25x applies
to the Rust path: a per-node contender index in front of the existing
tile loops, and the window computed per node from the contenders. The
screening constant c=8 gives a skipped runner a survival deficit of 6e-16
per node, times at most n skipped runners, so 6e-11 at n=100,000; c=7
would give 1e-7 and skip more.

**The tail.** A screened-out runner gets nothing at that node, so
runners that are never contenders come out exactly zero: 1,861 of 10,000
at 2^11 nodes, all with true probability below 1e-23. The engine never
returns zero, and a wide-window exact reference shows the engine is
right down to 1e-50 (its bulk window loses nothing for hopeless runners,
because the winner's upper tail dies faster than their density grows).
To avoid the zeros, the screened pass gets a "wavefront" tail: the
contender lattice already yields the winner distribution given f, so
take its mean and sd once per node, and give each skipped runner
Phi((m_W - M_j) / sqrt(sd_j^2 + s_W^2)), a two-horse race against the
effective winner, O(1) per skipped runner, accumulated in log space.
Cost +7% (8.2 s to 8.8 s at n=10,000, 2^11). It leaves every runner
above 1e-20 unchanged to 1% and removes all zeros. Below 1e-20 it
overstates, because the winner's upper tail is thinner than Gaussian:

| runner's exact p (2^8 nodes) | engine | screened + wavefront tail |
|---:|---:|---:|
| 1.0e-14 | 1.0e-14 | 1.0e-14 |
| 1.0e-18 | 1.0e-18 | 1.0e-18 |
| 1.0e-22 | 1.0e-22 | 6.7e-22 |
| 1.0e-26 | 1.0e-26 | 3.5e-24 |
| 1.0e-35 | 1.0e-35 | 7.1e-32 |
| 1.0e-50 | 1.0e-50 | 5.7e-43 |

Good to 1e-20, order-of-magnitude to 1e-26, and a positive number with
the right ordering beyond that, which is all a log-odds inverter needs
from a runner nobody has ever observed winning. If the far tail ever
mattered, the fix is to replace the Gaussian winner by a quadratic fit
to log g(x) at the top of the window and integrate against the runner's
Gaussian tail in closed form, still O(1) per skipped runner. Not done.

Rotation, adaptive stratification and composites are closed. Pruning is
subsumed by screening. Hybrid A (Parts 1 and 2) remains closed as a field
method.

## Tables

### Rotation and adaptive stratification, n=8, mean of max|p - truth| over 4 scrambles at 8192 passes

| field | Sobol 2^13 | rotate svd | rotate p-weighted | adaptive tree |
|---|---:|---:|---:|---:|
| rank 2 D 0.1 | 5.5e-05 | 5.5e-05 | 5.5e-05 | 1.3e-04 |
| rank 2 D 0.05 | 7.2e-05 | 5.8e-05 | 6.2e-05 | 1.5e-04 |
| rank 2 D 0.02 | 9.9e-05 | 6.8e-05 | 7.6e-05 | 1.7e-04 |
| rank 3 D 0.1 | 1.2e-04 | 1.5e-04 | 1.2e-04 | 6.1e-04 |
| rank 3 D 0.05 | 1.6e-04 | 2.0e-04 | 1.2e-04 | 1.3e-03 |
| rank 3 D 0.02 | 2.9e-04 | 3.3e-04 | 2.3e-04 | 4.1e-03 |
| rank 4 D 0.1 | 7.3e-04 | 3.6e-04 | 3.3e-04 (reverse control 3.0e-04) | 2.1e-03 |
| rank 4 D 0.05 | 8.5e-04 | 5.9e-04 | 3.9e-04 (control 4.3e-04) | 3.2e-03 |
| rank 4 D 0.02 | 1.1e-03 | 8.7e-04 | 5.0e-04 (control 6.0e-04) | 4.3e-03 |
| rank 5 D 0.1 | 6.4e-04 | 7.3e-04 | 5.8e-04 | 3.0e-03 |
| rank 5 D 0.05 | 8.5e-04 | 9.5e-04 | 9.6e-04 | 4.3e-03 |
| rank 5 D 0.02 | 1.1e-03 | 1.1e-03 | 1.4e-03 | 3.0e-03 |

### Composites, n=1,000, Sobol 2^11, 615 runners kept at margin 5 (their truth mass outside: 2.5e-10)

| method | runners per pass | kept max abs | targets max rel | group total max abs |
|---|---:|---:|---:|---:|
| prune (drop the rest) | 615 | 1.5e-03 | 1.2e-01 | 0 |
| k-means 5 composites | 620 | 2.6e-02 | 4.0e-01 | 5.5e-02 |
| k-means 20 composites | 635 | 4.6e-03 | 9.4e-02 | 3.8e-03 |
| seriation 5 composites | 620 | 5.3e-02 | 6.7e-01 | 9.7e-02 |
| seriation 20 composites | 635 | 5.0e-02 | 6.5e-01 | 4.0e-02 |
| full field | 1000 | 1.5e-03 | 1.2e-01 | |

### Pruning from a 2^7 pilot

| n | margin | kept | truth mass dropped | Sobol 2^13 all-n max abs | Sobol 2^15 |
|---|---:|---:|---:|---:|---:|
| 10,000 | 3 | 852 | 5.3e-05 | 4.1e-04 | 6.5e-05 |
| 10,000 | 5 | 3,893 | 6.1e-11 | 4.1e-04 | |
| 100,000 | 3 | 1,753 | 4.1e-04 | 2.2e-04 | 2.2e-04 |
| 100,000 | 5 | 14,896 | 8.5e-09 | 1.8e-04 | |

Full-field references: n=10,000 Sobol 2^13 4.1e-4, 2^15 5.3e-5; n=100,000
Sobol 2^13 1.8e-4, 2^15 1.1e-4.

### Per-node screening (c=8, 257 points), one process, 4 BLAS threads

| n | engine ms/pass | screened ms/pass | contenders per node | max diff to engine at 2^9 | Sobol 2^15 all-n max abs | 2^15 seconds |
|---|---:|---:|---:|---:|---:|---:|
| 1,000 | 4.2 | 1.1 | 146 | 6.5e-14 | 1.5e-04 | 36 |
| 10,000 | 44 | 4.4 | 530 | 2.0e-12 | 6.6e-05 | 140 |
| 100,000 | 448 | 18 | 1,630 | 3.3e-10 | 1.1e-04 | 591 |

The 2^15 errors agree with the engine's own 2^15 rows in Part 2 to within
the truth floor (n=10,000: 6.6e-5 here against 5.3e-5 there, floor
2.7e-5); the per-node window differs from the engine's shared window at
that level, not the estimator.
