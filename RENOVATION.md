# Renovation plan: `winning` becomes the ratings/applications layer on `thurstone`

*Status: executed in the working tree, July 2026 — nothing committed until reviewed.
Headline results (full tables in BENCHMARKS.md, exact-chain updater of July 9):
across thirteen dataset variants the lattice rater wins Formula 1 decisively and
sumo narrowly, sits in statistical ties atop WTA/ATP/EPL/Halo head-to-head, reaches
parity with TrueSkill on the drifting synthetic world and trails it slightly on the
rest of its home turf — with the best calibration (ECE) on chess and strong
distributional metrics throughout. The lab (planning/rating_lab.md) validated the
SIAM paper's Harville claim at scale and drove the exact-chain upgrade. Next research
steps: cavity handling in the update; market-implied per-horse scale via joint
win+place calibration (thurstone's 2-D path).*

The relationship mirrors `timemachines` / `skaters`: **thurstone** is the small, stable,
zero-frills core (lattice order statistics and the fast ability transform); **winning**
is reborn one layer up as the applications package — rating systems, benchmarking
against TrueSkill and friends, and eventually the other applied threads (racing
markets, rank forecasting, ability-transform statistics).

## 1. What is already in thurstone (nothing to move)

The entire core of old `winning` has been re-implemented, OO-style, in thurstone:

| old winning | thurstone |
|---|---|
| `lattice.py` density algebra (shifts, convolution, skew-normal, winner-of-many, multiplicity) | `lattice.py`, `density.py`, `order_stats.py` |
| `lattice_calibration.py` (`solve_for_implied_offsets` etc.) | `inference.AbilityCalibrator` |
| `std_calibration.py` / `skew_calibration.py` wrappers | subsumed by `AbilityCalibrator` + `STD_*`/`ALT_*` conventions |
| state price / dividend plumbing | `pricing.StatePricer`, `Race` |
| `normaldist.py` | `normaldist.py` |
| `lattice_conventions.py` | `conventions.py` |

Thurstone additionally has a multi-race layer old winning never had
(`GlobalAbilityCalibrator`, `GlobalLSCalibrator`, `KalmanAbilityTracker`,
`DynamicThurstoneCalibrator`, `MultiRayGlobalCalibrator`) — these are the fitters the
new winning wraps.

## 2. What stays in this repo (citation continuity)

- The SIAM paper PDFs (`docs/Horse_Race_Problem__SIAM_.pdf`, `..._updated.pdf`) and the
  full LaTeX source (`docs/latex_src/`).
- The BibTeX block in the README (Cotton 2021, SIAM J. Financial Mathematics 12(1),
  DOI 10.1137/19M1276261).
- `LITERATURE.md` (Thurstone references) and `docs/LITERATURE.md`.
- `examples_harville_comparison/` — reproduces tables from the paper (Harville and
  rule-of-a-quarter comparisons); moves to `papers/siam2021/` with its CSVs.
- `data/simple.csv` — 459,504 Betfair runners (bsp, finish_position, race_id across
  47,651 races; 34,092 races have both bsp and finish position for every runner).
  Uploaded 2021-11-03; anonymized (no names, dates, venues). Kept for planned
  market-calibration work; it lacks contestant identifiers, so the rating benchmarks
  use synthetic worlds and tennis instead.

## 3. What becomes thurstone issues/discussions (drafts in `planning/thurstone_issues/`)

Unported machinery and research ideas, each drafted as an issue for review before
posting — none are blockers for the ratings mission:

1. **Place/show/exotic pricing** (`lattice_simulation.py`: `skew_normal_place_pricing`,
   `placegetters_from_performances`, `exotic_count`, `longshot_adjusted_dividends`).
   Proposal: rank-k pricing belongs in thurstone core (it already owns winner-of-many);
   winning would shim it for racing applications.
2. **Gaussian copula / correlated contestants** (`lattice_copula.py`:
   `gaussian_copula_win`, `gaussian_copula_five`). Correlation via a common factor;
   used for M6. Candidate for a thurstone extension or a winning application module.
3. **Rank-k probabilities** (`five_prices_from_five_densities`) — the M6 primitive;
   natural companion of (1).
4. **Events / cumulative-scoring densities** (`densities_from_events`,
   `state_prices_from_events`) — in-play golf/tournament pricing from score + future
   scoring events.
5. **Ability transform as a statistical tool** (`Ability_Transforms_Updated.ipynb`) —
   proportions → winning probabilities → latent ability distributions, applied to ~30
   real datasets (market caps, city sizes, COVID counts, word frequencies…). Discussion
   piece: candidate for a thurstone docs page or a winning module + notebook revival.
6. **Turnout/participation copula** (`Democracy_Correlation_Trade.ipynb`) — coupling a
   Thurstonian preference to a correlated latent participation variable. Discussion.
7. **Pandas conveniences** (`pandas_util.py`) — groupby-apply ability transforms;
   preserved in `attic/` as a candidate future winning module (thurstone stays
   numpy-only).

## 4. What gets archived or deleted here

- Delete: old `winning/` package modules (all live on in thurstone), `testscolab/`,
  root `simple.csv` (duplicate of `data/simple.csv` with different anonymization),
  `Sepal_Width.ipynb` (near-identical duplicate of `Ability_Transforms_Updated.ipynb`),
  empty `requirements.txt`, `.DS_Store` files, seven of eight legacy workflow YAMLs.
- Archive (move, keep): notebooks → `attic/notebooks/`; the paper-reproduction
  examples → `papers/siam2021/` (pinned to winning 1.x — see its README); visual/scalability demos die (they demonstrate thurstone-core
  behaviour and belong there if anywhere).
- Old PyPI `winning` users: the README gains a pointer — core API moved to
  `thurstone`; `pip install winning==1.0.3` (the last *published* 1.x — the final
  1.x tree said 1.0.6 in setup.py but was never released) still gets the old
  package. The calibration import surface is preserved as deprecated shims
  (`std_calibration`, `skew_calibration`, `lattice_calibration`,
  `lattice_conventions`) with a parity test against real 1.0.3 output.

## 5. The new package

`src/` layout, `pyproject.toml` only, `dependencies = ["thurstone>=0.1.0"]` and nothing
else in core; extras: `benchmarks = ["trueskill", "openskill", "pandas"]`,
`dev = ["pytest", "ruff"]`.

```
src/winning/
    __init__.py          # small public surface
    ratingsystem.py      # the shared interface: observe(names, ranks) / rating / win_probabilities
    elo.py               # pure-python Elo (multi-entrant via Bradley-Terry softmax)
    glicko2.py           # pure-python Glicko-2 (pairwise decomposition for races)
    thurstonerating.py   # the new thing: outcome-driven Thurstonian rater
    exact.py             # exact win probabilities for any Gaussian-belief rater via thurstone lattice
    shims.py             # TrueSkill / OpenSkill wrapped into the RatingSystem interface (extras)
    benchmarks/
        events.py        # Event container + synthetic worlds with oracle truth
        forward_chain.py # elapse-predict-score-observe evaluation loop
        metrics.py       # log-loss, Brier, accuracy, Kendall tau
        tennis.py        # ATP tennis fetcher (N=2 real data)
        run_benchmark.py # CLI entry producing the README tables
tests/
```

### Design notes for `thurstonerating.py`

Thurstone's `KalmanAbilityTracker` is **price-driven** (inverts market prices). Rating
systems must learn from outcomes alone, so winning contributes the outcome-driven
rater:

- Belief per contestant: a **full density on the ability lattice** (not a
  Gaussian summary), with random-walk diffusion between events (convolution
  with N(0, tau^2 dt)); time advances via `elapse(dt)`.
- **Update from a finish order**: the exact likelihood of the whole order via
  an O(N) forward/backward chain on the lattice (opponents at predictive
  marginals), one pass; dead-heat events fall back to permutation-invariant
  group peeling. A two-runner update matches exact Bayes to lattice precision;
  the chain replaced Plackett peeling on July 9 after a lab test showed the
  peeled factorization was measurably lossy (research/exact_order_update.py).
- **Prediction**: exact win probabilities for the N-runner field via thurstone's
  winner-of-many on the lattice. This is the differentiator: TrueSkill and the
  Weng-Lin Thurstone-Mosteller variants decompose rankings into pairwise/adjacent
  terms; the lattice computes the joint order statistics exactly.
- Known limitation (measured): repeated pseudo-independent updates collapse
  posterior variance faster than TrueSkill's converged EP, costing probability
  calibration on long synthetic runs; proper cavity handling is future work.
- The same exact-prediction routine is offered to *all* Gaussian-belief systems in the
  benchmark (TrueSkill's and OpenSkill's (mu, sigma) beliefs are priced on the same
  lattice), so prediction quality is compared apples-to-apples, and separately from
  update-rule quality.

### Benchmark design

Forward-chained: for each race in time order, predict win probabilities (and finish
order) from current ratings, score, then update. Metrics: log-loss / Brier / accuracy
on the winner; Kendall tau / Spearman / NDCG on the full order. Baselines: uniform
(floor) and Betfair BSP implied probabilities (ceiling). Comparators: TrueSkill
(`trueskill`), OpenSkill PlackettLuce + ThurstoneMostellerFull (`openskill`), Glicko-2,
Elo. Horses are not identified across races in `simple.csv`, so the flagship runs use
synthetic populations with oracle ground truth — see `benchmarks/events.py` — plus
ATP tennis (Sackmann archive via a fork mirror, fetched not vendored, CC BY-NC-SA)
for the N=2 longitudinal case.

## 6. Open questions for review

- Post the seven issue drafts to microprediction/thurstone as issues, discussions, or
  one umbrella issue?
- Keep the `winning` name on PyPI for the renovated package (major-version bump to
  2.0.0), or is a rename on the table?
- The old package's users: worth a final 1.x release whose README points to thurstone,
  before 2.0.0 lands?
