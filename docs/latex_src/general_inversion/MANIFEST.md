# Manifest: provenance of the paper's tables and numbers

Paper: "Scalable Inversion of Contests with Correlated Performances,
Including Softmax and Multinomial Probit" (`main.tex` in this directory).

This file records the provenance of each table and each number in the
text: the command, seed and log line where a committed command produces
it, and otherwise where it is recorded. Every table has a command. The
last section lists the in-text measurements with no regenerating
command, and says which of them describe code since replaced.
`CLAIMS.md` records how the mappings changed through review; where the
two differ, this file is the current one.

## How to run

```sh
git clone https://github.com/microprediction/winning && cd winning
git checkout paper-r14      # paper-r12 plus this manifest and the files marked (r13) or (r14)
cd docs/latex_src/general_inversion
make env            # Python 3.12 virtualenv from requirements.lock (needs uv)
make <target>       # one row of the tables below; log in out/<target>.log
```

- Every command below is run from this directory unless a path is given.
  `$PY` is `../../../.venv/bin/python`. The `make` targets set one BLAS
  thread (`OMP_NUM_THREADS=1` and the others) before calling `$PY`.
- All scripts at `paper-r12` are unchanged in the revision, with one
  addition: `break.py` gained section I (a closed-form check of factor
  tail depth) on 2026-11-23; its other sections are as at the tag, and
  section I runs unchanged against the `paper-r12` package.
- Files added after `paper-r12` are marked **(r13)** or **(r14)** by the
  tag that first carries them: this manifest, `selfconv.py`,
  `research/general_sigma/paired_ensembles4.py` and
  `research/inversion_ablation/run_convergence_battery.py` (r13), and
  `equal_groups.py`, `small_figures.py`, `port_agreement.py`,
  `invert_structures.py` and `plot_inversion_scaling.py` (r14). All run unchanged against the `paper-r12`
  package. `paper-r14` is the tag of the final version.
- One data file changed at `paper-r14`: a stray line holding only the
  number 577697 was deleted from
  `research/general_sigma/results_ensembles4.csv` (line 602). No data row
  was touched. `summarize_ensembles4.py` stopped on that line.
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
| Equal groups: multiplicity, rank cut, TV against the block grammar (n=24) | 6.2 | **(r14)** `$PY equal_groups.py` | `default_rng(0)` for the abilities; pricing nodes are the pipeline's seeded default | one line per group count: multiplicity, bulk/lead eigenvalue ratio, rank cut, warning, fit rank, projected residual, TV. The four table columns are the first, second, fourth and last. Values regenerated at `paper-r14`: rows 3, 4, 6 and 12 moved in the last digit or two from the earlier print (2.8e-4, 1.8e-4, 1.0e-2, 4.0e-2 became 1.8e-4, 3.7e-4, 1.1e-2, 3.9e-2) |
| Raw minus identified, paired over 20 seeds, 15 ensembles (Table 2) | 6.4 | **(r13)** `$PY paired_ensembles4.py` in `research/general_sigma`; reads `results_ensembles4.csv`, which `make ensembles` writes | none in the analysis; the CSV's seeds are those of the next row | one line per ensemble: median difference, interval, wins, p, Holm |
| Ensemble accuracy: median, q90, worst TV and median per-entry error, 6 ensembles | 6.5 | `make ensembles` = `$PY run_ensembles4.py` in `research/general_sigma`, then `$PY summarize_ensembles4.py` | seed s = 0..19 per ensemble: covariance from `randomcov` at `1000 + s`; referee draws `default_rng(s)`; abilities `default_rng(5)`; pricing nodes Sobol `seed=3` | `summarize_ensembles4.py` prints the LaTeX rows (projected arm) |
| Kernel stratification: RBF and Matern at three length scales, three budgets | 6.6 | `make kernel` = `$PY run_kernel4.py` in `research/general_sigma` | s = 0..19: kernel inputs `default_rng(1000 + s)` via `kernel_C`; referee draws `default_rng(s)`; abilities `default_rng(5)` | one line per kernel, length scale and budget |
| Extrapolated GHK cost brackets (Appendix A) | A | `make law` = `$PY bench.py law` for the anchor (19.4 s at n=1000, R=1e3); `make scale` for the lattice column; the brackets are `194 s * (n/1000)^beta`, beta = 1.9 and 2.9, computed by hand | `default_rng(4)` (law), `default_rng(1)` (scale) | **time**: keys `law` in `retime2.json`, `scale` in `retime.json` |

## Numbers in the text

### Section 1

| Number | Command | Seed | Read |
|---|---|---|---|
| Forward rank-one factor at n=1e4, 1e5, 1e6: 0.12, 1.2, 18 s | `make scale` = `$PY bench.py scale` | `default_rng(1)` | **time**; key `scale` in `retime.json` |
| Inversion at n=1e6: under two minutes independent (105 s), about twelve minutes rank-one factor (699 s); log residuals 7.3e-10 and 7.7e-9; ability errors 7.5e-11 and 2.4e-9 | `make invertmillion` = `$PY bench.py invertmillion` | `default_rng(0)` per case | one line per n and class: "inversion ... max\|log p resid\| ... max\|mu err\|". **time** and the residuals as logged: key `invert_million` in `retime.json` |
| Inversion at n=1e6 for one field of each class: sweeps 11, 39 (independent, rank-one factor) and 30, 20, 30 (blocks, tree, nested); residuals 3.0e-9, 9.8e-9, 3.2e-9; ability errors 2.3e-10, 1.2e-9, 2.3e-10; smallest target probability 1.1e-26 or less in the last three | **(r14)** `$PY invert_structures.py 1000000` | `default_rng(0)` per class | one line per class: "inversion ... converged ... sweeps ... max\|log p resid\| ... max\|mu err\| ... min p". Logged run: `log_invert_structures.txt` (an M1 Ultra shared with other jobs, so its times are not the paper's; the nested class takes about an hour and a half there and all five about two and a half hours) |
| Documentation-site plot and table of inversion time against n, by class (Section 1.2, footnote); not printed in the paper | **(r14)** `$PY invert_structures.py N` for N = 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, output appended to `log_invert_structures.txt`; then `$PY plot_inversion_scaling.py` | `default_rng(0)` per class | `inversion_scaling.csv` and `inversion_scaling.png`. **time**; sweeps and residuals should reproduce, times depend on the machine |
| GHK over a hundred times slower at n=200, R=1e4 (1.84 s against 15 ms); about eleven times at R=1e3 (170 ms against 15 ms) | `make ghk` | as the GHK table | **time**; n=200 row |
| GHK log-log slope 1.9 from n=10 to 200 (R=1e4) and 2.9 from n=200 to 1000 (R=1e3); 0.18, 1.69, 19.4 s at n=200, 500, 1000 | `make law` = `$PY bench.py law`; the 1.9 slope is from the `ghk` table's R=1e4 times at n=10 and 200 | `default_rng(4)` | **time**; key `law` in `retime2.json` |
| Equicorrelation prices as independent noise, to 1e-16 (footnote) | **(r14)** `$PY small_figures.py G1` | `default_rng(0)` | "worst 1.1e-16" |

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
| Student-t, 4 degrees of freedom, six runners: level 1e-12 lies 1456 standard units out; spacing 5.7 at 257 points | **(r14)** `$PY small_figures.py G2` | none (closed form) | first G2 section |
| Student-t, nu=2.5, n=40: shipped window 2e-5 TV at 513 points and 6e-3 at the default 257; the 5e-3 under a normal envelope is earlier code (gap G2) | **(r14)** `$PY small_figures.py G2` | `default_rng(0)` | second G2 section: 5.6e-3, 2.0e-5, 1.1e-5 at 257, 513, 1025 points |
| Jacobian against finite differences: J 5e-10, J0 1e-9 (65-point Gaussian); both 1e-8 (Student-t, 257 points) | **(r14)** `$PY small_figures.py G3` | `default_rng(3)` | section G3a. Regenerated at `paper-r14`; the earlier print had 2e-11, 1e-8, 1e-6, 1e-5 from a configuration that was not recorded |
| Column sums 1e-16; row sums 1e-3 at 25 points, 2e-10 at 65, machine precision at 257 (Gaussian); 1e-3, 3e-8, machine precision (Gumbel) | **(r14)** `$PY small_figures.py G3` | `default_rng(5)` | section G3b |
| Removal example mu=(-20,0,1): 0.760; lattice mass 3e-28 on the original grid | partly: `$PY -m pytest -q ../../../tests/test_race_invariants.py::test_removal_shares_dominant_favorite` | none | asserts 0.760 to 1e-4; the 3e-28 mass on the original grid is not computed (gap G4) |

### Section 4

| Number | Command | Seed | Read |
|---|---|---|---|
| Tensor rule sizes 3,375; 50,625; 759,375 | arithmetic: 15^3, 15^4, 15^5 | none | |
| Raw row-norm example: 2.90, 4.10, 6.1e-3, 3.87 (footnote) | partly: `$PY -m pytest -q ../../../tests/test_bases.py::test_sharpness_dispatch_is_gauge_invariant_and_safe` | none | asserts the shipped dispatch is within 5e-5 of the analytic orthant; the four footnote figures describe the rejected rule and are not computed (gap G5) |
| Fixed 15-node rule loses up to five percent of total variation | `make fixed-rule` = `$PY fixed_rule_loss.py` in `research/fuzz` | case seed 11540 (`default_rng(seed)`) | "4.79e-2" for the fixed rule, "3.8e-3" for the default |
| Sharp-factor escalation is applied past a trigger of 3 | `make tests` (`tests/test_sharp_factor_escalation.py`) | seeds inside each test | pytest pass |
| Block kernel loses 1.0e-1 TV at cluster sharpness 18, against a four-million-path referee | **(r14)** `$PY small_figures.py G6` | field `default_rng(0)`; referee `default_rng(1)` | section G6. Regenerated at `paper-r14`; the earlier print had 5e-2 from a configuration that was not recorded |
| Rank two, about sixty nodes: tensor Gauss-Hermite 3e-5 against Sobol 3e-3 at n=50; 3e-6 against 3e-3 at n=200 | `make gh-vs-sobol` = `$PY run_gh_vs_sobol.py` in `research/factor_nodes` | `default_rng(24)`; Sobol scrambles `seed=s` | one line per n and rule |
| 400-runner cluster at correlation 0.99: proxy window drops 28 percent; shares 0.68/0.32 (footnote) | partly: `$PY -m pytest -q ../../../tests/test_blocks.py::test_window_covers_near_common_shock_winner_mass` | none | asserts shares of 0.5 under the shipped window; the 28 percent and 0.68/0.32 describe the rejected proxy window, which is no longer in the code (gap G5) |
| Depth-one tree with zero strengths reproduces the block race to machine precision (footnote) | partly: `$PY -m pytest -q ../../../tests/test_polish.py::test_tree_jacobian_flat_equals_block` | `default_rng(3)` | asserts the Jacobians agree to 1e-12; for probabilities `research/fuzz/fuzz_races.py` (`make fuzz`) checks `tree_flat_is_blk` to 5e-9 on every case |
| Tree recursion against a 2^22-path Monte Carlo: root-mean-square z between 0.75 and 1.1, depths one to four | `make tree` = `$PY run_exactness.py` in `research/tree_exactness` | `default_rng(2)` | "rms z" per depth |
| Tree tails against minimax tilting: 0.3 percent at probabilities down to 7e-8 | `make botev-tree` = `Rscript botev_tree.R` in `research/tree_exactness` (needs R, TruncatedNormal) | `set.seed(1)` | ratios 1.002 to 1.003 |
| Cophenetic identity of the Remark | `make tests` (`tests/test_race_invariants.py::test_cophenetic_tree_race_identity`) | seed inside the test | pytest pass |

### Section 5

| Number | Command | Seed | Read |
|---|---|---|---|
| Directional derivative against central differences: 3e-2 without the quotient and 2e-2 with it at L=25; 1e-11 either way from L=101; asymmetry 3e-4 at 25 points, 7e-11 at 65 | **(r14)** `$PY small_figures.py G3` | `default_rng(5)` for the field, `default_rng(7)` for the direction | sections G3c and G3b. The L=25 figures were regenerated at `paper-r14`; the earlier print had 2e-3 and 6e-4 from a configuration that was not recorded |
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
| Equal-groups table (n=24, correlation 0.7); eigenvalue ratios 0.051, 0.034; fit rank 9 and projected residuals 0.07, 0.10, 0.32 for six, eight and twelve groups | **(r14)** `$PY equal_groups.py` | `default_rng(0)` | columns "bulk/lead", "fit rank", "proj resid" |
| Dense n=2000: fit and price 3.6 s (1.0 s fit) against 18.5 s for 1e6 draws; 82 percent of the 1,563 alternatives priced below 1e-5 take no simulation win; bulk TV 1.01e-2 | `make large-n` = `$PY run_large_n.py` in `research/general_sigma` | `default_rng(7)`; nodes `seed=3`; draws `seed=9` | summary lines. **time**: `timings_2026-11-11/README.md`, `quiet_large_n.log` |
| Split referee at n=2000: 63.6 percent zero-win; model mass 1.13e-4; 111 wins; interval [9.13e-5, 1.34e-4]; TV 4.4e-3 against a floor of 5.4e-3 | `make split-referee` = `$PY run_split_referee.py --n 2000 --draws 2000000` in `research/general_sigma` | covariance `default_rng(7)`; halves `seed=11`, `seed=12` | summary lines; also `split_referee_2000.json` |
| Identified against raw arm: better on 11 of 15 unadjusted, 10 after Holm; sparse precision 5.1e-3 against 6.8e-3; block equicorrelation 2.5e-3 against 2.0e-2, worst seed 3.2e-3 against 0.16 | **(r13)** `$PY paired_ensembles4.py` for the counts; `$PY summarize_ensembles4.py` for the medians and worst seeds, both in `research/general_sigma` | as Table 2 | last line of `paired_ensembles4.py`; rows `block_equicorr`, `sparse_precision` of the summary |
| Archakov-Hansen median TV 3.7e-2 | `make ensembles`, then `summarize_ensembles4.py` | as the ensemble table | row `archakov_hansen`, projected arm |
| Kernel text: rank-27 eigenfit, projected residual 1e-3; 2^16 nodes reach 6e-3; greedy allocation residual 0.14; up to 7e-2 TV | none (gap G8) | | recorded in `research/general_sigma/NOTES.md`, a comment in `winning/factor/core.py` and `CLAIMS.md`; `run_kernel4.py` does not run 2^16 nodes or report the rank |

### Section 7

| Number | Command | Seed | Read |
|---|---|---|---|
| Shared field against per-alternative quadrature at n=200, rank 2: TV 3e-15; 17.5 s against 14 ms (1,270 times) | `make bm` = `$PY bench.py bm` | `default_rng(0)` | single line. **time**: key `per_alternative` in `retime.json` |
| One Genz probability at n=200: 0.57 s at maxpts 2.5e5, within its error estimate 2e-4; full vector 114 s against 14 ms | `make genz200` = `$PY bench.py genz200` (calls R's mvtnorm) | `default_rng(4)`; R `set.seed(20260827)` | **time**: key `genz200` in `retime2.json` |
| Self-convergence on the GHK-table problems: 3e-8 to 3e-7 TV; at most 1e-7 in a probability and 2.3e-5 (below 3e-5) in a log probability; smallest probability 2e-12 | **(r13)** `$PY selfconv.py` | `default_rng(4)`, the draw order of `bench.py ghk` | one line per n |
| Referee agreement: mvtnorm 2.2e-7 against a bound of 7.1e-7 (n=10); 1.3e-8 against 6.7e-8 at smallest probability 3e-10; tilting 1.0e-4 relative against a reported 4.6e-4; adaptive quadrature 1.9e-15; invariance battery 1e-16; two-runner 7e-9 | `make referee` = `$PY referee_cases.py && $PY referee_check.py` (replays `referee_out.json`); `make referee-r` reruns R (mvtnorm 1.4-2, TruncatedNormal 2.3) | cases `default_rng(4)`; R `set.seed(20260827)` | `referee_check.py` prints one line per case and referee |
| Stress boundaries: 4e-13 at p near 1e-30; 2e-8 at 1e-50; breakdown near 1e-119; round trip 2e-9 in log with a 1e-10 entry; Gumbel equals softmax to 1e-15 at n=1000; duplicates split exactly; Gauss-Hermite 25 nodes 8e-3 at loading scale 4; Sobol 4e-4 | `make stress` = `$PY break.py` | `default_rng(99)`; Sobol `seed=1` | sections A (vanishing noise), C (deep tails), D (Gumbel), E (duplicates), F (round trip), G (Jacobian-vector product), H (factor strength) |
| Factor tail depth, two-runner closed form: the default fifteen-node rule is within 0.2 percent at p near 5e-10 and low by a factor of twelve at 1e-20; an inversion of 1e-20 on that rule converges to the wrong contrast | **(r13)** `make stress` = `$PY break.py`, section I | none | lines "a=15" (closed/rule 0.9988) and "a=22.688" (12.35) |
| Python and R escalated rules agree to 3e-4 on a loading-scale-4 case | **(r14)** `$PY port_agreement.py` (needs Rscript and jsonlite) | `default_rng(99)`; reference Sobol `seed=1` | "TV Python default against R default 2.99e-04". The earlier print was 5e-4 (5.15e-4 in a commit message, case not recorded); the paper now carries the regenerated figure |
| Share inversion: RMSE over the sixteen alternatives with true share above one percent ("identified alternatives: 16/30"); smallest alternative p=2.5e-5, expected count 1.27; 13 of 40 replications empty; paired differences 0.0112 +/- 0.0006 (18.7 standard errors, 39 of 40) and 0.0017 +/- 0.0002 (6.9 standard errors, 33 of 40) | `make estimation` | as the table | "(seed ..., k empty)" per replication; "paired against exact" |
| Covariate fit: compiled forward kernel about six percent of the exact path's time | profile recorded in `CLAIMS.md` (reproduction map, covariates row); no make target | n/a | not regenerated by a command |
| Fishing data: exact likelihood still rising at loading norms near 4000 | none (gap G10) | | recorded in the docstring of `winning/mnprobit.py`; the Fishing data is not in the repository |
| Genz-Bretz arm: 31 vector evaluations at 3.15 s is 98 s, about thirty times the 3.0 s fit | arithmetic on the against-the-field table (n=30 Genz time) and the share-inversion table (fit time) | none | |
| Ten million contestants in 102 s (block grammar, ten thousand clusters, 257 points) | `make tenmillion` = `$PY bench.py tenmillion` (about 5 GB) | `default_rng(1)` | **time**: key `tenmillion` in `retime.json` |
| Rust classic calibration 100 to 200 times the pure-Python speed, n=200 down to 10 | `$PY bench.py classic` (run by `make timings`, key `classic`) | `default_rng(3)` | **time**: `retime3.json` (94 to 207 times) |
| Parity: 22 scenarios; R and JavaScript against the Python reference, nineteen to 1e-13 or better and three optimizer-mediated to 2e-6 or better (tolerances 5e-4 to 5e-3); Rust against NumPy to 1e-10 or better | `Rscript ../../../parity/check.R` and `node ../../../parity/check.mjs` replay `parity/vectors.json`, which `$PY ../../../parity/gen_vectors.py` writes; `$PY -m pytest -q ../../../tests/test_rust_parity.py` | inputs `default_rng(2026)` in `gen_vectors.py` (`make_inputs`); no random numbers in the three optimizer-mediated scenarios; `default_rng(7)` in the Rust test | one "ok/FAIL name max\|diff\| (tol)" line per scenario. Largest of the nineteen: 3.6e-14 (R), 1.4e-14 (JavaScript). The three: 1.7e-7, 1.1e-6, 2.1e-7 (R); 1.7e-7, 1.1e-6, 1.7e-7 (JavaScript). The JavaScript replay of the tree projection takes about four minutes |

## Numbers with no regenerating command

Listed here so that the gaps are visible in one place.

These in-text figures were measured during development and are recorded
in the places named, but no committed command prints them. Group G5 and
the items marked "earlier code" in G2 and G4 describe code since
replaced and cannot be produced by the shipped package; the others
could be scripted. Six groups were closed at `paper-r14`: G7 (the
equal-groups table) by `equal_groups.py`; G1, G3 and G6 and most of G2
by `small_figures.py`; G9 by `port_agreement.py`; G12 (sweep counts) by
`invert_structures.py`.
Where a regenerated figure differs from the earlier print, the row above
says so and the paper carries the regenerated figure. Where
a committed test pins the same configuration at a looser tolerance, the
rows above name it.

| Gap | Paper location | Figures | Status |
|---|---|---|---|
| G2 | Section 3.1 | 5e-3 TV under a normal envelope for the window | earlier code |
| G4 | Section 3.3, footnote | lattice mass 3e-28 | no script; removal on the winner's grid is not what the shipped routine does (earlier code) |
| G5 | Sections 4.1, 4.2, 6.1 | figures describing rules the code no longer contains: raw row-norm dispatch (2.90, 4.10, 6.1e-3, 3.87), proxy window (28 percent, 0.68/0.32), raw-residual fitter (1.7e-2) | cannot be regenerated from the shipped code; tests pin the shipped behavior |
| G8 | Section 6.6 | rank 27, residual 1e-3; 2^16 nodes reach 6e-3; 0.14; 7e-2 | no script |
| G10 | Section 7 | Fishing likelihood rising at loading norms near 4000 | no script; data not in the repository |
| G11 | Section 7 | compiled kernel about six percent of the covariate fit's time | a profile, recorded in `CLAIMS.md`; no command |
