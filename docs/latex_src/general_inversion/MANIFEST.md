# Manifest: every table and number in the paper, its command and its seed

Paper: "Scalable Inversion of Contests with Correlated Performances,
Including Softmax and Multinomial Probit" (`main.tex` in this directory).

This file maps each table and each number in the text to the command
that produces it, the seed that command uses, and the log line to read.
`CLAIMS.md` records how the mappings changed through review; where the
two differ, this file is the current one.

## How to run

```sh
git clone https://github.com/microprediction/winning && cd winning
git checkout paper-r12
cd docs/latex_src/general_inversion
make env            # Python 3.12 virtualenv from requirements.lock (needs uv)
make <target>       # one row of the tables below; log in out/<target>.log
```

- Every command below is run from this directory unless a path is given.
  `$PY` is `../../../.venv/bin/python`. The `make` targets set one BLAS
  thread (`OMP_NUM_THREADS=1` and the others) before calling `$PY`.
- All scripts at `paper-r12` are unchanged in the revision. Four files
  were added by the revision and are not in `paper-r12`: this manifest,
  `selfconv.py`, `research/general_sigma/paired_ensembles4.py` and
  `research/inversion_ablation/run_convergence_battery.py`. They are
  marked **(r13)** and are in the revision tag `paper-r13`. The scripts
  run unchanged against the `paper-r12` package.
- "Seed" is the argument of the NumPy generator (`np.random.default_rng`)
  that draws the problem, or R's `set.seed`. "None" means the computation
  uses no random numbers.
- Wall-clock figures are marked **time**. They are re-measured by
  `make timings` (`retime.py`), and the logs behind the paper's values
  are in `timings_2026-11-11/`. Times are machine-dependent; every other
  figure regenerates on the pinned environment, apart from the two
  sources of variation listed in `REPRODUCE.md`.

## Tables

| Table | Section | Command | Seed | Read |
|---|---|---|---|---|
| The covariance grammar (Table 1) | 1.2 | none: a list of models | n/a | n/a |
| Against the field: lattice, Genz, frequency, Mendell-Elston, Clark at n=10, 30 | 7 | `make alt` = `$PY bench.py alt` | problems `default_rng(4)`; QMC reference Sobol `seed=1`; frequency draws `default_rng([5, n])`; SciPy Genz unseeded by its API | one line per method per n: time, TV, tail. **time** columns and the two time-matched frequency rows: median of three quiet runs (`REPRODUCE.md`) |
| Against GHK: lattice, GHK R=1e3, GHK R=1e4 at n=10, 50, 200 | 7 | `make ghk` = `$PY bench.py ghk` | problems `default_rng(4)`; QMC reference Sobol `seed=1`; GHK draws: fixed per-alternative offsets inside `fastrace` | one line per n: time, TV, tail for each arm. **time** columns from `timings_2026-11-11/retime.json`, key `ghk` |
| Share inversion under sampling noise: RMSE and paired differences, 40 replications | 7 | `make estimation` = `$PY run_mle2.py --reps 40 --out out/results_40.json` in `research/mnp_estimation` | design `default_rng(3)`; replication r uses `default_rng(100 + r)`, r = 0..39 | block "method / median rmse / mean +/- se" and "paired against exact". **time** column: key `saturated` in `retime.json` (first eight replications) |
| Covariate estimation: RMSE of beta, 8 replications | 7 | `make covariates` = `$PY run_mle3.py && $PY run_mle4.py` in `research/mnp_estimation` | design `default_rng(9)`; replication r uses `default_rng(1000 + r)` | summary block. **time** column: key `covariates` in `retime2.json` |
| Equal groups: multiplicity, rank cut, TV against the block grammar (n=24) | 6.2 | partly: `tests/test_fit_covariance.py::test_group_degeneracy_is_predictable_and_detected` (see Section 6.2 below) | `default_rng(0)` | gap G7 |
| Raw minus identified, paired over 20 seeds, 15 ensembles (Table 2) | 6.4 | **(r13)** `$PY paired_ensembles4.py` in `research/general_sigma`; reads `results_ensembles4.csv`, which `make ensembles` writes | none in the analysis; the CSV's seeds are those of the next row | one line per ensemble: median difference, interval, wins, p, Holm |
| Ensemble accuracy: median, q90, worst TV and median per-entry error, 6 ensembles | 6.5 | `make ensembles` = `$PY run_ensembles4.py` in `research/general_sigma`, then `$PY summarize_ensembles4.py` | seed s = 0..19 per ensemble: covariance from `randomcov` at `1000 + s`; referee draws `default_rng(s)`; abilities `default_rng(5)`; pricing nodes Sobol `seed=3` | `summarize_ensembles4.py` prints the LaTeX rows (projected arm). The committed `results_ensembles4.csv` has one malformed line (602) on which that script stops; `make ensembles` rewrites the file, and `paired_ensembles4.py` skips the line |
| Kernel stratification: RBF and Matern at three length scales, three budgets | 6.6 | `make kernel` = `$PY run_kernel4.py` in `research/general_sigma` | s = 0..19: kernel inputs `default_rng(1000 + s)` via `kernel_C`; referee draws `default_rng(s)`; abilities `default_rng(5)` | one line per kernel, length scale and budget |
| Extrapolated GHK cost brackets (Appendix A) | A | `make law` = `$PY bench.py law` for the anchor (19.4 s at n=1000, R=1e3); `make scale` for the lattice column; the brackets are `194 s * (n/1000)^beta`, beta = 1.9 and 2.9, computed by hand | `default_rng(4)` (law), `default_rng(1)` (scale) | **time**: keys `law` in `retime2.json`, `scale` in `retime.json` |

## Numbers in the text

### Abstract and Section 1

| Number | Command | Seed | Read |
|---|---|---|---|
| Forward rank-one factor at n=1e4, 1e5, 1e6: 0.12, 1.2, 18 s | `make scale` = `$PY bench.py scale` | `default_rng(1)` | **time**; key `scale` in `retime.json` |
| Inversion at n=1e6: under two minutes independent (105 s), about twelve minutes rank-one factor (699 s); log residuals 7.3e-10 and 7.7e-9; ability errors 7.5e-11 and 2.4e-9 | `make invertmillion` = `$PY bench.py invertmillion` | `default_rng(0)` per case | one line per n and class: "inversion ... max\|log p resid\| ... max\|mu err\|". **time** and the residuals as logged: key `invert_million` in `retime.json` |
| GHK over a hundred times slower at n=200, R=1e4 (1.84 s against 15 ms); about eleven times at R=1e3 (170 ms against 15 ms) | `make ghk` | as the GHK table | **time**; n=200 row |
| GHK log-log slope 1.9 from n=10 to 200 (R=1e4) and 2.9 from n=200 to 1000 (R=1e3); 0.18, 1.69, 19.4 s at n=200, 500, 1000 | `make law` = `$PY bench.py law`; the 1.9 slope is from the `ghk` table's R=1e4 times at n=10 and 200 | `default_rng(4)` | **time**; key `law` in `retime2.json` |
| Equicorrelation prices as independent noise, to 6e-17 (footnote) | partly: `$PY -m pytest -q ../../../tests/test_race_invariants.py::test_case_v_is_choice_equivalent_to_independence` | `default_rng(0)` | the test asserts agreement below 1e-12; the 6e-17 itself is not printed (gap G1) |

### Section 2

| Number | Command | Seed | Read |
|---|---|---|---|
| Genz full vector at n=30 costs nearly nine hundred times the lattice | `make alt` | as the table | ratio of the n=30 times, 3.15 s / 3.6 ms |
| Frequency simulation: 11 to 28 times the TV at matched time, tail log-errors near 0.6, 4 to 11 times at ten times the budget | `make alt` | as the table | ratios of the table's TV cells |
| Mendell-Elston 2e-3 at n=10, 3e-2 at n=30, tail off by more than two; Clark tail off by nearly four | `make alt` | as the table | table cells; e^0.82 and e^1.36 |
| Minimax tilting all-n: n=50, 0.3 s against 0.07 s, error 2.5e-3 against 2.9e-5; n=200, 31 s against 0.28 s, 1.4e-3 against 4.1e-5; R=1e4 at n=200, 50 s | `$PY run_tilting.py` in `research/experiments/exp22_minimax_tilting` (also run by `make timings`, key `tilting`) | `SEED = 21`; references `seed=301`, `302` | lines "N=50 R=...: err ..., ...s" and "N=50 lattice: err ...". **time**: key `tilting` in `retime2.json` |

### Section 3

| Number | Command | Seed | Read |
|---|---|---|---|
| Bulk window: 33 points match 500 on the span window; span floor 6e-11; five or more orders at 33 to 65 points, two to eight times from 129 | `make window` = `$PY run_window_savings.py` in `research/lattice_window` | `default_rng(0)` | table of error by point count for the two windows |
| Student-t, 4 degrees of freedom: tail level 1e-12 lies 1456 standard units out; spacing 5.7 at 257 points | none (gap G2) | | recorded in the docstring of `winning/factor/races.py` |
| Student-t, nu=2.5, n=40: 5e-3 TV under a normal envelope, 5e-5 with the true survival | none (gap G2) | | recorded in `CLAIMS.md` (5.1e-3 before, 5.2e-5 after) and the same docstring |
| Jacobian against finite differences: J 2e-11, J0 1e-8 (65-point Gaussian); 1e-6, 1e-5 (Student-t) | partly: `$PY -m pytest -q ../../../tests/test_polish.py::test_jacobian_is_the_derivative_of_the_map_that_was_evaluated` | `default_rng(3)` | asserts J below 1e-9 (Gaussian, 65 points) and below 5e-6 (Student-t, at 257 points); the J0 figures are not computed (gap G3) |
| Column sums 6e-17; row sums 1e-3 at 25 points, 2e-10 at 65, machine precision at 257 | partly: `$PY -m pytest -q ../../../tests/test_polish.py::test_jacobian_matches_finite_differences` | `default_rng(5)` | asserts row sums below 1e-9 at 257 points; the column-sum figure and the 25- and 65-point row sums are not computed (gap G3) |
| Removal example mu=(-20,0,1): 0.760; lattice mass 3e-28 on the original grid | partly: `$PY -m pytest -q ../../../tests/test_race_invariants.py::test_removal_shares_dominant_favorite` | none | asserts 0.760 to 1e-4; the 3e-28 mass on the original grid is not computed (gap G4) |

### Section 4

| Number | Command | Seed | Read |
|---|---|---|---|
| Tensor rule sizes 3,375; 50,625; 759,375 | arithmetic: 15^3, 15^4, 15^5 | none | |
| Raw row-norm example: 2.90, 4.10, 6.1e-3, 3.87 (footnote) | partly: `$PY -m pytest -q ../../../tests/test_bases.py::test_sharpness_dispatch_is_gauge_invariant_and_safe` | none | asserts the shipped dispatch is within 5e-5 of the analytic orthant; the four footnote figures describe the rejected rule and are not computed (gap G5) |
| Fixed 15-node rule loses up to five percent of total variation | `make fixed-rule` = `$PY fixed_rule_loss.py` in `research/fuzz` | case seed 11540 (`default_rng(seed)`) | "4.79e-2" for the fixed rule, "3.8e-3" for the default |
| Sharp-factor escalation is applied past a trigger of 3 | `make tests` (`tests/test_sharp_factor_escalation.py`) | seeds inside each test | pytest pass |
| Block, nested, tree kernels lose 5e-2 TV at cluster sharpness 18, against a four-million-path referee | partly: `$PY -m pytest -q ../../../tests/test_blocks.py::test_sharp_blocks_warn_that_gh_cannot_converge` | none | asserts the warning fires; the 5e-2 and its four-million-path referee have no script (gap G6) |
| Rank two, about sixty nodes: tensor Gauss-Hermite 3e-5 against Sobol 3e-3 at n=50; 3e-6 against 3e-3 at n=200 | `make gh-vs-sobol` = `$PY run_gh_vs_sobol.py` in `research/factor_nodes` | `default_rng(24)`; Sobol scrambles `seed=s` | one line per n and rule |
| 400-runner cluster at correlation 0.99: proxy window drops 28 percent; shares 0.68/0.32 (footnote) | partly: `$PY -m pytest -q ../../../tests/test_blocks.py::test_window_covers_near_common_shock_winner_mass` | none | asserts shares of 0.5 under the shipped window; the 28 percent and 0.68/0.32 describe the rejected proxy window, which is no longer in the code (gap G5) |
| Depth-one tree with zero strengths reproduces the block race to machine precision (footnote) | partly: `$PY -m pytest -q ../../../tests/test_polish.py::test_tree_jacobian_flat_equals_block` | `default_rng(3)` | asserts the Jacobians agree to 1e-12; for probabilities `research/fuzz/fuzz_races.py` (`make fuzz`) checks `tree_flat_is_blk` to 5e-9 on every case |
| Tree recursion against a 2^22-path Monte Carlo: root-mean-square z between 0.75 and 1.1, depths one to four | `make tree` = `$PY run_exactness.py` in `research/tree_exactness` | `default_rng(2)` | "rms z" per depth |
| Tree tails against minimax tilting: 0.3 percent at probabilities down to 7e-8 | `make botev-tree` = `Rscript botev_tree.R` in `research/tree_exactness` (needs R, TruncatedNormal) | `set.seed(1)` | ratios 1.002 to 1.003 |
| Cophenetic identity of the Remark | `make tests` (`tests/test_race_invariants.py::test_cophenetic_tree_race_identity`) | seed inside the test | pytest pass |

### Section 5

| Number | Command | Seed | Read |
|---|---|---|---|
| Directional derivative against central differences: 2e-3 at L=25 without the quotient; 3e-11 from L=101; 6e-4 at L=25 (grid motion); asymmetry 3e-4 at 25 points, 7e-11 at 65 | none (gap G3) | | 2.06e-3, 3.1e-11 and 6.2e-4 are recorded in `CLAIMS.md`; the asymmetry figures are recorded nowhere else |
| Own-slope iteration 16, 17, 27 sweeps; unpreconditioned map fails in 500 sweeps on two fields and needs 68 to 144 on the third (n=200, rank two) | `make blp-ablation` = `$PY run_blp_ablation.py` in `research/inversion_ablation` | `default_rng(31)` | one line per field and map |
| Stall case: damped iteration stalls near 1e-4 (n=200, loadings N(0,1.6^2), variances in [0.15,0.25]); Newton-Krylov converges to 9.6e-10 on the benchmark field at about sixty times the wall clock and diverges on the stall case | `make newton-krylov` = `$PY run_newton_cg3.py` in `research/experiments/exp23_newton_cg` | `default_rng(21)` | "Jacobi stalls at 1.1e-4"; "9.6e-10". The sixty-fold is a **time** ratio in the same log |
| Naive Newton-CG: 370 s without converging against 3.5 s (footnote) | `$PY run_newton_cg.py` in `research/experiments/exp23_newton_cg` (run by `make timings`, key `newton_cg`) | `SEED = 21` | **time**; `retime3.json` |
| Convergence battery at n=100: 125 of 130 converge; median sweeps 10 to 21 at scale 0.25 and 53 to 58 at scale 2; five rank-one failures at scale 2, residual at most 9.5e-8 | **(r13)** `$PY run_convergence_battery.py` in `research/inversion_ablation`; log `log_convergence_battery.txt` | cell (grammar g, scale j, replicate r): `default_rng([2026, g, j, r])` | one line per cell |
| Round trips at n=400: independent 0.05 s, blocks 0.17 s, tree 0.20 s, factor rank two 0.57 s, nested 1.18 s; log residuals below 1e-8 | `make invert` = `$PY bench.py invert` | `default_rng(3)` | one line per grammar. **time**: key `grammar_inversion` in `retime.json` (residuals 3.7e-9 to 8.7e-9) |
| Mislocation under an assumed independent model: 2.49 against a spread of 8.24, median 0.34 (n=200, thirty clusters, correlation 0.65) | `make mislocate` = `$PY bench.py mislocate` | `default_rng(7)` | last line |

### Section 6

| Number | Command | Seed | Read |
|---|---|---|---|
| Two-alternative example: difference variance 0.5 against 1.4 (footnote) | arithmetic: s=(1,2), rho=0.9: 1+4-2(0.9)(2)=1.4; (1-rho)(1+4)=0.5 | none | |
| Raw-residual variant distorts an in-grammar input by 1.7e-2 TV; projected pipeline at the 2e-4 node-noise floor | partly: `$PY -m pytest -q ../../../tests/test_fit_covariance.py::test_in_grammar_truth_recovered` | `default_rng(0)` | asserts the shipped pipeline below 1e-3; 1.7e-2 describes the rejected raw-residual variant (gap G5) |
| Absolute floor on diag(1e-8,1e-8,1): Var(X1-X2) from 2e-8 to 6.7e-4, choice error 0.48; pairwise diagnostic 3.3e4; global residual 5e-4; below 1e-8 after the fix | `$PY -m pytest -q ../../../tests/test_fit_covariance.py::test_relative_floor_preserves_near_singular_contrasts` | none | the test rebuilds the failure by hand: `_worst_contrast_ratio` returns 33299 and the global norm 5.5e-4; the shipped fit's `contrast_residual_max` is 6.9e-10 |
| In-grammar n=12 stall at objective 0.88, rescued to 1e-13 | `$PY -m pytest -q ../../../tests/test_fit_covariance.py::test_in_grammar_small_n_spread_variances_fits_exactly` | `default_rng(0)` | pytest pass |
| Multistart at n=300, eight starts: objectives agree to 4e-16; choices agree to 1e-14 on three ensembles; block equicorrelation differs by 0.25 TV; 8e-4 at the tie's rank | `make multistart` = `$PY run_multistart.py --starts 8 --n 300` in `research/general_sigma` | ensembles `default_rng(0)`; starts `default_rng(11)` | one block per ensemble |
| Equal-groups table (n=24, correlation 0.7) and eigenvalue ratios 0.051, 0.034 | partly: `$PY -m pytest -q ../../../tests/test_fit_covariance.py::test_group_degeneracy_is_predictable_and_detected` | `default_rng(0)` | asserts the pattern for 2, 4, 6 and 12 groups (TV below 1e-3 or above 5e-3, multiplicity g-1); the six table values and the two eigenvalue ratios are not printed (gap G7) |
| Dense n=2000: fit and price 3.6 s (1.0 s fit) against 18.5 s for 1e6 draws; 82 percent zero-win alternatives; bulk TV 1.01e-2 | `make large-n` = `$PY run_large_n.py` in `research/general_sigma` | `default_rng(7)`; nodes `seed=3`; draws `seed=9` | summary lines. **time**: `timings_2026-11-11/README.md`, `quiet_large_n.log` |
| Split referee at n=2000: 63.6 percent zero-win; model mass 1.13e-4; 111 wins; interval [9.13e-5, 1.34e-4]; TV 4.4e-3 against a floor of 5.4e-3 | `make split-referee` = `$PY run_split_referee.py --n 2000 --draws 2000000` in `research/general_sigma` | covariance `default_rng(7)`; halves `seed=11`, `seed=12` | summary lines; also `split_referee_2000.json` |
| Identified against raw arm: better on 11 of 15 unadjusted, 10 after Holm; sparse precision 5.1e-3 against 6.8e-3; block equicorrelation 2.5e-3 against 2.0e-2, worst seed 3.2e-3 against 0.16 | **(r13)** `$PY paired_ensembles4.py` for the counts; `$PY summarize_ensembles4.py` for the medians and worst seeds, both in `research/general_sigma` | as Table 2 | last line of `paired_ensembles4.py`; rows `block_equicorr`, `sparse_precision` of the summary |
| Archakov-Hansen median TV 3.7e-2 | `make ensembles`, then `summarize_ensembles4.py` | as the ensemble table | row `archakov_hansen`, projected arm |
| Kernel text: rank-27 eigenfit, projected residual 1e-3; 2^16 nodes reach 6e-3; greedy allocation residual 0.14; up to 7e-2 TV | none (gap G8) | | recorded in `research/general_sigma/NOTES.md`, a comment in `winning/factor/core.py` and `CLAIMS.md`; `run_kernel4.py` does not run 2^16 nodes or report the rank |

### Section 7

| Number | Command | Seed | Read |
|---|---|---|---|
| Shared field against per-alternative quadrature at n=200, rank 2: TV 3e-15; 17.5 s against 14 ms (1,270 times) | `make bm` = `$PY bench.py bm` | `default_rng(0)` | single line. **time**: key `per_alternative` in `retime.json` |
| One Genz probability at n=200: 0.57 s at maxpts 2.5e5, within its error estimate 2e-4; full vector 114 s against 14 ms | `make genz200` = `$PY bench.py genz200` (calls R's mvtnorm) | `default_rng(4)`; R `set.seed(20260827)` | **time**: key `genz200` in `retime2.json` |
| Self-convergence on the GHK-table problems: 3e-8 to 3e-7 TV; at most 1e-7 in a probability and 2e-5 in a log probability; smallest probability 2e-12 | **(r13)** `$PY selfconv.py` | `default_rng(4)`, the draw order of `bench.py ghk` | one line per n |
| Referee agreement: mvtnorm 2.2e-7 against a bound of 7.1e-7 (n=10); 1.3e-8 against 6.7e-8 at smallest probability 3e-10; tilting 1.0e-4 relative against a reported 4.6e-4; adaptive quadrature 1.9e-15; invariance battery 1e-16; two-runner 7e-9 | `make referee` = `$PY referee_cases.py && $PY referee_check.py` (replays `referee_out.json`); `make referee-r` reruns R (mvtnorm 1.4-2, TruncatedNormal 2.3) | cases `default_rng(4)`; R `set.seed(20260827)` | `referee_check.py` prints one line per case and referee |
| Stress boundaries: 4e-13 at p near 1e-30; 2e-8 at 1e-50; breakdown near 1e-119; round trip 2e-9 in log with a 1e-10 entry; Gumbel equals softmax to 1e-15 at n=1000; duplicates split exactly; Gauss-Hermite 25 nodes 8e-3 at loading scale 4; Sobol 4e-4 | `make stress` = `$PY break.py` | `default_rng(99)`; Sobol `seed=1` | sections A (vanishing noise), C (deep tails), D (Gumbel), E (duplicates), F (round trip), G (Jacobian-vector product), H (factor strength) |
| Python and R escalated rules agree to 5e-4 on the worst case | none (gap G9) | | recorded in a commit message (5.15e-4) |
| Share inversion: smallest alternative p=2.5e-5, expected count 1.27; 13 of 40 replications empty; paired differences 0.0112 +/- 0.0006 (18.7 standard errors, 39 of 40) and 0.0017 +/- 0.0002 (6.9 standard errors, 33 of 40) | `make estimation` | as the table | "(seed ..., k empty)" per replication; "paired against exact" |
| Covariate fit: compiled forward kernel about six percent of the exact path's time | profile recorded in `CLAIMS.md` (reproduction map, covariates row); no make target | n/a | not regenerated by a command |
| Fishing data: exact likelihood still rising at loading norms near 4000 | none (gap G10) | | recorded in the docstring of `winning/mnprobit.py`; the Fishing data is not in the repository |
| Genz-Bretz arm: 31 vector evaluations at 3.15 s is 98 s, about thirty times the 3.0 s fit | arithmetic on the against-the-field table (n=30 Genz time) and the share-inversion table (fit time) | none | |
| Ten million contestants in 102 s (block grammar, ten thousand clusters, 257 points) | `make tenmillion` = `$PY bench.py tenmillion` (about 5 GB) | `default_rng(1)` | **time**: key `tenmillion` in `retime.json` |
| Rust classic calibration 100 to 200 times the pure-Python speed, n=200 down to 10 | `$PY bench.py classic` (run by `make timings`, key `classic`) | `default_rng(3)` | **time**: `retime3.json` (94 to 207 times) |
| Parity: 22 scenarios, 19 within 1e-7, three optimizer-mediated at 5e-4 to 5e-3 | `$PY ../../../parity/gen_vectors.py` writes `parity/vectors.json`; `Rscript parity/check.R` and `node parity/check.mjs` replay it; the R package's `testthat` golden files run in CI | inputs `default_rng(2026)` in `gen_vectors.py` (`make_inputs`); no random numbers in the three optimizer-mediated scenarios | one "ok/FAIL name max\|diff\| (tol)" line per scenario; tolerances are the third argument of each `sc(...)` call |

## Numbers with no regenerating command

Listed here so that the gaps are visible in one place.

These figures were measured during development and are recorded in the
places named, but no committed command at `paper-r12` prints them. Where
a committed test pins the same configuration at a looser tolerance, the
rows above name it.

| Gap | Paper location | Figures | Status |
|---|---|---|---|
| G1 | Section 1.1, footnote | 6e-17 | a test asserts 1e-12; the tighter figure is not printed |
| G2 | Section 3.1 | Student-t window: 1456, 5.7; 5e-3 against 5e-5 | no script |
| G3 | Sections 3.2 and 5.2 | Jacobian against finite differences (J0 figures; 25- and 65-point row sums; column sums; quotient 2e-3, 3e-11, 6e-4; asymmetry 3e-4, 7e-11) | tests assert looser bounds at other lattice sizes; no script prints these |
| G4 | Section 3.3, footnote | lattice mass 3e-28 | no script |
| G5 | Sections 4.1, 4.2, 6.1 | figures describing rules the code no longer contains: raw row-norm dispatch (2.90, 4.10, 6.1e-3, 3.87), proxy window (28 percent, 0.68/0.32), raw-residual fitter (1.7e-2) | cannot be regenerated from the shipped code; tests pin the shipped behavior |
| G6 | Sections 4.1, 4.2 | 5e-2 TV at cluster sharpness 18 against a four-million-path referee | a test asserts the warning; no referee script |
| G7 | Section 6.2 | equal-groups table values; 0.051, 0.034 | a test asserts the pattern for four of the six rows; values not printed |
| G8 | Section 6.6 | rank 27, residual 1e-3; 2^16 nodes reach 6e-3; 0.14; 7e-2 | no script |
| G9 | Section 7 | Python and R escalated rules agree to 5e-4 | no script |
| G10 | Section 7 | Fishing likelihood rising at loading norms near 4000 | no script; data not in the repository |
| G11 | Section 7 | compiled kernel about six percent of the covariate fit's time | a profile, recorded in `CLAIMS.md`; no command |
| G12 | Section 7 | million-contestant inversion sweep counts | not logged by `bench.py invertmillion` |
