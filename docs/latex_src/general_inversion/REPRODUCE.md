# Reproducing "Scalable Inversion of Contests with Correlated Performances"

Every number in `main.tex` is produced by a committed, seeded script in this
repository. `CLAIMS.md` maps each claim to its `make` target and script.

## One command

```sh
git clone https://github.com/microprediction/winning && cd winning
git checkout paper-r12
cd docs/latex_src/general_inversion
make env      # Python 3.12 virtualenv at the repository root (needs uv)
make          # the accuracy claims, about 15 minutes
```

`make full` adds the long accuracy runs (ensemble battery, kernel table,
estimation tables, the R referees, the large-n benchmarks): several hours.
`make timings` re-measures every wall-clock figure; see below.

Logs land in `out/<target>.log`; each target also prints its last lines.

## Environment

- **Python.** `requirements.lock` pins every package (Python 3.12). The
  `winning` package itself is this repository at the tag, installed
  editable; `fastrace` 0.1.2 supplies the compiled kernels used by the
  GHK comparator and the large-n benchmarks.
- **R** (only for `make referee-r` and `make botev-tree`). Measured with
  R 4.6.1, mvtnorm 1.4.2, TruncatedNormal 2.3, jsonlite 2.0.0. The default
  `make referee` replays the committed R output (`referee_out.json`) instead.
- **Threads.** Every target runs with one BLAS thread
  (`OMP_NUM_THREADS=1` and friends) so results do not depend on core count.

## Determinism

- All scripts seed NumPy generators explicitly; the R referees call
  `set.seed`. Accuracy figures regenerate exactly on the pinned environment,
  apart from the two sources of variation below.
- **Wall-clock-matched estimators.** The "frequency, matched time" rows of
  the against-the-field table draw as many samples as fit in the lattice's
  run time, so their errors vary with machine speed. They draw from their
  own random stream, so nothing else in the table depends on them.
- **Genz in SciPy** (`bench.py alt`) is randomized internally and unseeded
  by its API; its TV agrees with the lattice at the reference's noise in
  every run, but the last digit moves. (`genz200` calls R's `mvtnorm`, which
  is seeded.)

## Timings

Times are from one Apple Mac Studio (M3 Ultra, 28 cores, 96 GB), single-threaded BLAS; the lattice kernels use one core and the Rust GHK comparator uses all free cores. The paper's
timing claims are ratios (lattice against GHK, Genz, simulation); absolute
seconds will differ on other hardware. `make timings` runs `retime.py`, which
re-measures every wall-clock figure, prints it beside the paper's value, and
refuses to measure on a machine whose load average is high (`--force`
overrides and says so in the output). The logs behind every time in the
paper are in `timings_2026-11-11/`.

## Known limits

- `fuzz_races.py` reports three `hopeless_others` flags (largest 3.4e-4) on
  rank-1 fields at sharpness about 4.4, where Gauss-Hermite's factor
  quadrature error is itself of that size (checked against a 120-node,
  40,001-point referee). Adding a hopeless runner re-centres the loadings and
  so changes the node order; both answers sit within the factor-strength
  boundary the paper states. The flag is the battery's 1e-5 tolerance, not a
  claim of the paper.
- The fixed-15-node "loses up to 5%" figure was found by the fuzz battery
  before the adaptive rule existed (`research/fuzz/log_fuzz2000.txt`).
  `make fixed-rule` reproduces it on the shipped code by forcing the old
  fixed rule: 4.79e-2 TV at seed 11540, against 3.8e-3 for the default.
