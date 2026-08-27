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
| Genz n=200: 4.6 s/probability, agreement to 6 digits | commit ed3f411 snippet; fold into `bench.py` |
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

## Gaps (claims whose committed backing is thin)

1. Rust scale paragraph: "ten million contestants in 64 seconds ...
   232x the pure-Python implementation" is documented in
   `rust/winning/README.md` / `lib.rs` but has no committed seeded
   bench script. Action: commit the bench harness or soften the claim.
2. "Inverting block-generated probabilities under an assumed
   independent model mislocates abilities by up to a fifth of the
   field's spread": mapping not yet pinned to a specific script
   (candidates in `research/general_sigma/`). Action: pin or re-measure
   in `break.py`.
3. The n=200 Genz single-probability measurement lives in a commit
   message snippet; fold into `bench.py alt` as a flagged slow case.
