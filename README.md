# winning — rating systems on the thurstone ability transform

[![CI](https://github.com/microprediction/winning/workflows/CI/badge.svg)](https://github.com/microprediction/winning/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Rating systems for multi-entrant contests — races, tournaments, leaderboards —
built on the exact lattice order statistics of the
[thurstone](https://github.com/microprediction/thurstone) package, and
benchmarked against TrueSkill, OpenSkill, Glicko-2 and Elo.

`thurstone` is the small, stable core: densities on a lattice, winner-of-many,
and the fast ability transform. This package is the applications layer, the way
[timemachines](https://github.com/microprediction/timemachines) sits on
[skaters](https://github.com/microprediction/skaters): cores are few and
stable; applications multiply, so they get their own package.

## Install

    pip install winning            # core: depends only on thurstone (numpy)
    pip install winning[benchmarks]  # + trueskill, openskill, pandas comparators

## Quick start

```python
from winning import ThurstoneRating

tr = ThurstoneRating()
tr.observe(names=["ada", "ben", "cid", "dot"], ranks=[1, 2, 3, 4])
tr.observe(names=["ada", "cid", "eve"], ranks=[1, 2, 3])

tr.win_probabilities(["ada", "cid", "eve"])   # exact field win probabilities
tr.rating("ada")                              # Rating(mu=..., sigma=...)
tr.leaderboard()                              # best-first, conservative
```

Every system in the package speaks the same three verbs — `observe(names,
ranks)`, `win_probabilities(names)`, `rating(name)` — including the shims
around third-party comparators (`winning.shims.TrueSkillRating`,
`winning.shims.OpenSkillRating`).

## What is different about ThurstoneRating

- **Beliefs are whole densities, not (mu, sigma) pairs.** Each contestant's
  ability belief lives on the thurstone lattice and is free to be skewed or
  multimodal; updates multiply in exact per-stage likelihoods of the observed
  finish order (Plackett peeling of the joint order statistics) rather than
  pairwise Gaussian approximations. A single update matches exact Bayes to
  lattice precision (see `tests/test_thurstonerating.py`).
- **Predictions are exact.** Field win probabilities come from the lattice
  winner-of-many computation — dead-heat aware, O(N) in field size — not from
  pairwise decompositions. The same routine is exposed as
  `winning.gaussian_win_probabilities(mus, sigmas, beta)` so any
  Gaussian-belief system (TrueSkill included) can be priced exactly, which
  separates update quality from prediction-formula quality in benchmarks.
- **Race-native.** Full finish orders of N-entrant fields are the primary
  input, not an afterthought bolted onto a 1v1 system.

## Benchmarks (measured, not asserted)

Twelve datasets, prequential evaluation (predict each event before observing
it), with market ceilings and oracle floors where they exist. Full tables,
metric definitions, dataset provenance and licensing in
[BENCHMARKS.md](BENCHMARKS.md); reproduce any row with
`python -m winning.benchmarks.run_benchmark --dataset <name>`.

| Dataset | Best by log loss | ThurstoneRating (this package) |
|---|---|---|
| Formula 1, 1,158 GPs 1950-2026 | **ThurstoneRating** | wins log loss, accuracy, tau and rank-PIT |
| WTA tennis, 31k matches | **ThurstoneRating** | wins every metric |
| ATP tennis, 32k matches | Elo (by 0.0009) | 2nd; best Brier, accuracy, tau |
| Chess, 121k Lichess games | site's own ratings, then TrueSkill | best calibration of any system (ECE 0.0057) |
| Sumo, 110k bouts | Glicko-2 | 4th in a pack spanning 0.006 |
| EPL football, 7.6k matches | Bet365 odds, then Elo | 2nd system, 0.002 behind Elo |
| HK horse racing, 6.3k races | pari-mutuel odds, then Glicko-2 | 2nd system, ahead of TrueSkill |
| Halo 2 head-to-head (TrueSkill's data) | four-way tie | in the tie |
| Halo 2 free-for-all (TrueSkill's data) | TrueSkill | clear 2nd; best rank-PIT among leaders |
| Synthetic races (x3 worlds, oracle floor) | TrueSkill (its own generative model) | 2nd; best rank-PIT (0.0069 vs 0.0196) |

Three patterns, measured across the suite:

1. **Full finish orders are where the lattice wins.** The two outright wins
   (F1, WTA) and the near-ties come wherever joint order statistics matter;
   pairwise-decomposition systems visibly degrade as fields grow (Glicko-2
   falls below uniform on 20+ car F1 grids; OpenSkill's ThurstoneMosteller
   diverges on Halo free-for-all).
2. **Calibration is the consistent edge.** Best or near-best ECE and rank-PIT
   almost everywhere — the predicted *distributions* of finish positions match
   reality, not just the favorites.
3. **Markets remain the ceiling** wherever they exist (Bet365, pari-mutuel,
   Lichess's own ratings) — they price information no outcome-only rating
   system sees. Closing that gap is the applications agenda of this package.

The heteroskedastic synthetic world (per-contestant noise) degrades every
system — all incumbents assume common performance noise. Per-contestant scale
learning, which thurstone's 2-D (loc, scale) calibration supports, is the
designated next step and no benchmarked system offers it.

TrueSkill is patented by Microsoft and licensed for non-commercial use; it
appears here strictly as a research comparator. OpenSkill exists precisely to
be the unencumbered alternative, as does this package.

## The market-calibration application

`data/simple.csv` holds 459,504 Betfair runners (starting price, finish
position, anonymized race id across 47,651 races), kept for planned work on
how well market-implied abilities forecast outcomes — the ceiling any rating
system chases. No code in the package reads it yet. The ability transform that
converts win prices to relative abilities in one shot is
`thurstone.AbilityCalibrator`; see the
[thurstone docs](https://github.com/microprediction/thurstone).

## Where everything went (the 2.0 renovation)

Versions 1.x of this package contained the original implementation of the
SIAM-paper algorithm. That core now lives in, and should be imported from,
[thurstone](https://github.com/microprediction/thurstone); `pip install
winning==1.0.3` (the last published 1.x) still gets the old package, and the
most-used 1.x imports (`winning.std_calibration`, `winning.skew_calibration`,
`winning.lattice_calibration`, `winning.lattice_conventions`) are preserved in
2.x as deprecated shims that delegate to thurstone — verified to reproduce
winning 1.0.3 numbers to lattice precision (see `tests/test_legacy.py`). The migration map is in
[RENOVATION.md](RENOVATION.md); unported application ideas are preserved in
[attic/](attic) with draft upstream issues in
[planning/thurstone_issues/](planning/thurstone_issues); the paper PDFs, LaTeX
source and table reproductions remain in [docs/](docs) and
[papers/siam2021/](papers/siam2021).

## Cite

    @article{doi:10.1137/19M1276261,
    author = {Cotton, Peter},
    title = {Inferring Relative Ability from Winning Probability in Multientrant Contests},
    journal = {SIAM Journal on Financial Mathematics},
    volume = {12},
    number = {1},
    pages = {295-317},
    year = {2021},
    doi = {10.1137/19M1276261},
    URL = {https://doi.org/10.1137/19M1276261}
    }
