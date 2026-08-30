# Claim-to-script manifest

Every empirical claim in main.tex, mapped to the seeded script or test
that produces it. The title footnote promises this; this file is the
index. Paths are repo-relative.

## Beside the paper (this directory)

| Claim | Source |
|---|---|
| GHK table: 200x / 5x at n=200, times and TV | `bench.py ghk` |
| GHK cost law n^2.8 points | `bench.py law` |
| Scale: 0.18 s / 2.7 s / 29 s at 1e4..1e6 | `bench.py scale` |
| Against-the-field table (Genz, frequency, ME, Clark) | `bench.py alt` |
| Genz n=200 single probability: 0.43 s at maxpts=250k, agreement to 4-5 digits (six digits needs multi-second budgets) | `bench.py genz200` (pinned 2026-08-27; supersedes an earlier 4.6 s / six-digit note taken at a different tolerance) |
| Referee agreement (mvtnorm within own bound; Botev 2.2e-4 at 3e-10; quadrature 1.9e-15; invariance battery) | `referee_cases.py` + `referee.R` + `referee_check.py` |
| Stress boundaries (tails to 1e-119, round-trip 1e-9, Gumbel=softmax 1e-15, duplicates, JVP 8e-12, sharpness escalation) | `break.py` |

## Elsewhere in the repo

| Claim | Source |
|---|---|
| Bulk window: 33 points beat 500, span floor 6e-11, 100x on hopeless fields | `research/lattice_window/run_window_savings.py` |
| Fixed 15-node rule silently loses up to 5% (sharp factors) | `research/fuzz/fuzz_races.py` (+ logs) |
| Sharp-factor family escalation pinned | `tests/test_sharp_factor_escalation.py` |
| GH tensor beats Sobol at rank 2, wide margin | `research/experiments/exp24_factor_rqmc/` |
| Depth-one tree reproduces block race; common root invisible | `tests/test_race_invariants.py` |
| Tree vs 2^22 CRN referee (RMS z 0.75-1.1, depths 1-4); Botev 0.3% to 7e-8 | `research/tree_exactness/run_exactness.py` |
| HRP cophenetic identity (floored) | `tests/test_race_invariants.py::test_cophenetic_tree_race_identity`, `research/hrp_finish/run_finish.py` |
| Jacobian off-diagonal symmetry/signs, zero row sums | `tests/test_race_invariants.py` |
| Inversion to 1e-8; sub-resolution floored | `break.py` section F; `research/experiments/exp35_independent_inversion/` (methodologically independent reference) |
| Ensemble accuracy table (1-4e-5 on 12/15; 1-4e-4 dense-strong; 4e-3 Matern/RBF) | `research/general_sigma/run_ensembles.py`, `run_ensembles2.py` (+ logs) |
| Dense n=2000: 5 s vs 36 s, 82% zero-win tails priced | `research/general_sigma/log_large_n*.txt` (script: `run_cv.py` family) |
| Estimation table (exact MLE 0.0187/4.9 s vs MSL) | `research/mnp_estimation/run_mle.py` |
| Parity: 22 scenarios, four languages | `r/winning/tests/testthat/` golden files + counterparts per port |

## Gaps: CLOSED (2026-08-27)

1. Rust scale: `bench.py tenmillion` committed and run on a quiet
   machine: 244.9 s at n=1e7, 1e4 clusters, 257-point lattice
   (streaming field). The research-note 64 s figure was a different
   configuration; the paper now cites the pinned 245 s.
2. Mislocation: `bench.py mislocate` pins it. Measured: max 2.49 vs
   spread 8.24 = 30.2% (median 0.34); paper updated from "a fifth" to
   "three tenths" with the configuration stated.
3. Genz n=200 single probability: `bench.py genz200` pins it. Measured:
   0.43 s per probability at maxpts=250k with estimated error 2e-4
   (agreement with the lattice to ~4-5 digits); the lattice prices all
   200 in 25 ms. Six-digit agreement needs multi-second maxpts budgets.

## Ninth review (referee 3), pinned measurements (2026-08-28)

1. Identified-objective fit: `factor_model_projected` minimizes
   min ||P(Sigma - VV' - diag D)P||_F (ALS; D-step collapsed to its
   n-dim normal equations, Gram exactly P∘P — 67 s → 0.27 s at n=300,
   identical minimizer to 1e-14). Packaged one-call:
   `fit_covariance` / `race_probabilities(cov=)`. In-grammar truths
   return at the 2e-4 node-noise floor (the raw-residual pipeline
   scored 1.7e-2 — the bug the ninth review's point 5 flushed out).
2. Multi-seed ensemble battery: `research/general_sigma/run_ensembles4.py`,
   20 seeds per ensemble, n=300, 1M-draw MC referee per seed, both arms
   (raw eigenfit pipeline vs identified pipeline), kernel stratified
   RBF/Matérn-3/2 × length scale {0.08, 0.2, 0.4} × promoted rank
   {5, 12}. randomcov pinned at commit 0d27a51.
3. Per-grammar inversion round trips: `bench.py invert`, n=400.
   independent 0.04 s, tree 0.37 s, blocks 0.67 s, factor r2 1.7 s,
   nested 3.7 s; max log-probability residual < 1e-8 in every grammar
   through the shipped `abilities_from_race(structure=)`.
4. Per-alternative factor-quadrature comparator (Butler–Moffitt run
   per alternative): `bench.py bm`, n=200 rank 2. Agrees with the
   shared field to TV 3e-15 and max log-ratio 3e-10; costs 21.9 s
   against 36 ms — 601×. Same conditioning, no shared field.
5. Cross-language fitter parity: R `fit_covariance` matches python's
   fitted model covariance VV'+diag(D) to 3e-15 on a mixed factor/AR
   test matrix (priced probabilities differ TV 2e-3: Halton vs Sobol
   nodes, by design).
6. Kernel stratification (final, two-arm fitter): med TV over 20 seeds —
   RBF ls 0.08/0.2/0.4 at m=5: 3.4/2.6/1.6e-2 (m=12: 3.1/2.6/1.7e-2;
   m=5 @2^14 nodes: 3.3/1.2/0.82e-2); Matérn-3/2: 2.9/2.5/2.1e-2
   (m=12: 2.5/2.3/1.7e-2; @2^14: 2.4/1.6/1.2e-2). Two regimes: short
   scale = representation (rank helps, nodes don't), long scale =
   quadrature (nodes help, rank doesn't; 2^16 reaches 6e-3 on the
   diagnostic draw). The old kernel row (up to 7e-2) was greedy rank
   misallocation, repaired by the eigen arm (stage vi).
7. n=2000 one-call: fit 0.6 s + price 3.6 s (fit was 41 s before the
   double-centering/subspace/water-filling rework, commit db111e3;
   NNLS closed form verified to 1e-14 vs scipy).

## Inversion at a million (2026-08-29)

`bench.py invertmillion`. The paper's headline inversion benchmark is
N=1e4 "in under a minute"; that was the table's limit, not the
method's. Measured on one laptop:

| n | structure | forward | inversion | max abs mu error |
|---|---|---|---|---|
| 1e5 | independent | 0.7 s | 6.8 s | 1.2e-11 |
| 1e5 | factor rank-1 | 3.4 s | 111.6 s | 2.8e-09 |
| 1e6 | independent | 7.7 s | 80.3 s | 7.5e-11 |
| 1e6 | factor rank-1 | 31.2 s | 1319.9 s | 2.6e-09 |

So the per-operation scale statement is now: forward block field 1e7,
forward rank-one factor 1e6, INVERSION 1e6 (80 s independent, 22 min
rank-one factor), full rank-two factor inversion 1e4 in under a minute.
Dense covariance remains a separate claim because it must first be
fitted into a scalable grammar.
