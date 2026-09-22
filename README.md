# winning

*A package for dealing with races, correlated or not.*

**Documentation, live demos, and the papers: [winning.microprediction.org](https://winning.microprediction.org).**
The [convergence demo](https://winning.microprediction.org/converge.html)
runs the lattice race against GHK and Mendell–Elston in your browser.

A race has contestants with abilities and a joint distribution of
performances. `race_probabilities` prices every contestant's chance of
winning in one pass over a shared field, for Gaussian, Gumbel, or any
other base, and `abilities_from_race` inverts observed probabilities
back to abilities. Both accept the same covariance descriptions: factor
loadings (`V=`, `D=`), a named grammar (`structure=` independent,
factor, blocks, nested, tree), or a dense `cov=` that is fitted to the
grammar on the way in.

[![CI](https://github.com/microprediction/winning/workflows/CI/badge.svg)](https://github.com/microprediction/winning/actions)
[![Julia](https://github.com/microprediction/winning/actions/workflows/julia.yml/badge.svg)](https://github.com/microprediction/winning/actions/workflows/julia.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![The fastest time, the best offer and the most popular product are the same order statistic](docs/assets/images/winning-pic.png)

## Install

    pip install winning        # core depends only on numpy and scipy

## Quick start

Independent race: shares from abilities, and back.

```python
import numpy as np
from winning import race_probabilities, calibrate_abilities

mu = np.array([-0.5, 0.0, 0.2, 0.3])          # lower is better (min-wins)
p = race_probabilities(mu)                     # array([0.443, 0.232, 0.175, 0.150])
mu_back = calibrate_abilities(p)               # recovers mu (mean zero)
```

Correlated race: a hundred runners moved by two common factors, all
shares in one pass over a shared survival field, then inverted.

```python
rng = np.random.default_rng(0)
N, k = 100, 2
mu = rng.normal(0, 1, N); mu -= mu.mean()
V = rng.normal(0, 0.4, (N, k))                # factor loadings
D = rng.uniform(0.5, 1.5, N)                  # idiosyncratic variances

p = race_probabilities(mu, V=V, D=D)          # all N shares, O(QNL)
mu_hat = calibrate_abilities(p, V=V, D=D)     # inversion
```

Counterfactuals and structure from the same shared field:

```python
from winning import removal_shares, tie_densities

q = removal_shares(mu, V=V, D=D)   # q[i][j] = P(j wins | i removed)
w = tie_densities(mu, V=V, D=D)    # photo-finish weights: the Jacobian's
                                   # graph-Laplacian (circuit) conductances
```

The probit conventions. `winning.probit` takes utilities, with higher
better, and returns shares. It carries both of the paper's
calibrations: utilities from observed shares, and the factor structure
itself from a supplied covariance.

```python
from winning.probit import shares, utilities_from_shares, fit_factor_model

utilities = -mu                            # higher is better on this side
p = shares(utilities, V=V, D=D)            # all N choice probabilities
u = utilities_from_shares(p, V=V, D=D)     # the paper's calibration
Sigma = V @ V.T + np.diag(D)
V_hat, D_hat = fit_factor_model(Sigma, k=2)  # certified rank-k contrast fit
p2 = shares(utilities, Sigma=Sigma, k=2)   # same fit applied en route
```

Distribution and correlation are chosen per call, and factor probit is
one named point in the family.

```python
from winning.factor import race_probabilities

race_probabilities(mu)                       # the classic independent race
race_probabilities(mu, V=V, D=D)             # factor probit
race_probabilities(mu, base="gumbel")        # Luce / softmax, exactly
race_probabilities(mu, V=V, base="gumbel")   # correlated softmax
race_probabilities(mu, temperature=0.7)      # E[softmin(X/tau)]: soft credit
```

Temperature is exact. By the Gumbel-argmin identity the softmin
expectation equals the hard race with each base convolved with the
tau-scaled Gumbel kernel, so the same engine serves it. Inversion holds
the temperature fixed; a single race pins down the abilities alone.

Custom densities go in as formulas: any standardized survival and
density pair, skewed or multimodal, is a `base=` callable. Empirical
data go to `winning.classic`: histograms, integer scores, and dead
heats with real mass are lattice atoms there, and its multiplicity
calculus prices ties exactly. Formulas to `base=`, atoms to `classic`;
the module docstrings give the measured cost of crossing over.

## What is in the package

- `winning.factor`: the engine. The all-share forward pass, inversion,
  exact Jacobians and tie densities, covariance fitting
  (`fit_covariance`), and constrained polish.

- `winning.factor` also prices finishing positions: top-k membership at
  any depth (`top_k_probabilities`, `rank_probabilities`), calibration
  of those curves (`abilities_from_topk`), joint location and scale per
  runner from the win and place curves (`loc_scale_from_win_and_second`),
  ordered-prefix probabilities (`ordered_probabilities`), and the
  Plackett-Luce likelihood (`winning.factor.permutations`).

- `winning.probit`: the same machine in the probit literature's
  max-wins, utilities-and-shares conventions.

- `winning.classic`: the original SIAM-paper lattice ability transform,
  in racing vocabulary (dividends, state prices, dead heats).

- `winning.ratings`: dynamic and factor ratings on the same engine,
  described next.

- `winning.methods` and `winning.bench`: every rival method behind one
  interface, and a seeded accuracy-time benchmark grid,
  `python -m winning.bench.runner`.

- `winning.research`: the density-agnostic research engine for
  arbitrary bases.

## Ratings

`winning.ratings` turns contest results into ratings and predictions
with the race engine underneath, so the model that is fitted is the
model that is priced.

`AbilityTracker` keeps one belief per entity and updates it from a
finishing order, a winner, scores, or market prices, at a cost
proportional to the entrants of each contest. `rate_history` and
`walk_forward` carry a full joint covariance instead and score held-out
predictions. Finishing orders are passed best first, and
`order_from_positions`, `order_from_performance` and `order_from_times`
build one from whatever form the data arrive in.

`fit_factor_ratings` estimates a level plus offsets over conditions
observed on the contest, such as time control or task category, with
one ridge penalty per covariate: the level pools every contest and the
offsets shrink toward it. On Lichess it beats the site's own
per-time-control Glicko-2 by 0.008 to 0.012 nats per game across three
splits with both sides tuned, and on both Lichess and Chatbot Arena it
beats a single rating and a separate rating per category. `sweep_offset_ridge` tunes the
penalty, and `covariate_contrast_report` tells you whether a covariate
was assigned or chosen, which decides how much a null result means.

Block correlation, `AbilityTracker(rho=..., groups=...)`, gives
same-group entrants a shared per-contest shock, fitted and priced under
one model. `tune_block_rho` selects the share by the filter's own
evidence and can sweep rho and the noise scale together, since the two
trade off under an order likelihood. On Formula 1 the team correlation
is real and changes from season to season, so tune it on a recent
window; the [ratings
page](https://winning.microprediction.org/ratings.html) has the
measurements.

`python -m winning.ratings.verify` runs the identities and calibration
checks the engine is held to, from a fast profile in CI to a full
nightly one.

A second ratings line, with whole-density beliefs and exact
full-finish-order updates, is being integrated from [src/](src) and
will ship as `ThurstoneRating`. Its benchmarks against TrueSkill,
OpenSkill, Glicko-2 and Elo on twelve datasets are in
[BENCHMARKS.md](BENCHMARKS.md): a decisive win on Formula 1 over 1,158
grands prix, the best calibration on chess (ECE 0.0047), statistical
ties at the top on WTA, ATP and the EPL, and markets as the ceiling
wherever they exist.

## The papers

Six manuscript projects live here, indexed with venue status in
[papers/README.md](papers/README.md).

The correlated calibration is documented in *Scalable Share Calibration
for Factor Multinomial Probit Models*
([papers/factor-probit-transform](papers/factor-probit-transform)): all
shares of a correlated Gaussian race in one O(QNL) pass, matrix-free
graph-Laplacian derivatives, and inversion at ten thousand alternatives
in under a minute. Every number comes from a committed, seeded script in
[research/experiments](research/experiments), and
`research/experiments/run_all_paper.py` regenerates the lot.

## Demos and other languages

[research/demos](research/demos) holds explanatory scripts: the shared
survival field, the cavity downdate. [js/factor](js/factor) is a
dependency-free JavaScript port at machine-precision parity with the
Python, for browser demos. [r/winning](r/winning) is a pure-R package.
[rust/fastrace](rust/fastrace) holds the optional compiled kernels,
built with `pip install maturin && maturin develop --release`, and
`winning.methods` uses them automatically.

[julia/winning](julia/winning) covers the factor races and the full
top-k module, pinned to the Python reference by `Pkg.test` and
`julia parity/check.jl`; the covariance grammar and classic lattice
are next on its roadmap.

Three Julia packages build on it. Two are in the General registry,
`] add MultinomialProbit` and `] add GMRFExtremes`.
[julia/MultinomialProbit](julia/MultinomialProbit) is the first
multinomial probit for Julia, with exact likelihood, analytic score,
and a common-random-numbers GHK for head-to-heads.
[julia/GMRFExtremes](julia/GMRFExtremes) answers order-statistic
queries on Gauss-Markov chains, argmax marginals, max CDF, first
passage and expected maximum, and takes a chain from
GaussianMarkovRandomFields.jl through its precision matrix.
[julia/FactorMvNormalCDF](julia/FactorMvNormalCDF) computes
deterministic MVN rectangle probabilities for factor covariance and
hands every other covariance to MvNormalCDF.jl, which it depends on;
its registration is in progress.

## History

Versions 1.x were the SIAM paper's reference implementation. That API
lives on as `winning.classic`, maintained, rust-accelerated, and
parity-locked against the R and JavaScript ports, and the old import
paths such as `winning.lattice_calibration` still work, pointing at the
new home with a `DeprecationWarning`.

The 2.0 renovation built a density-agnostic engine as a separate
`thurstone` package. It lives here as `winning.research`, the retired
research engine: kept for the pipelines built on it, but for
calibration use `calibrate_abilities`, which is materially faster and
more accurate and uses the compiled kernels. The `winning.thurstone`
alias is gone; the external `thurstone` shim must import
`winning.research` directly. The renovation's
migration notes are preserved in [planning/](planning) and
[attic/](attic).

## Cite

For the correlated engine (the shared field, the covariance grammars,
the substitution Jacobian, removal counterfactuals, and inversion at
scale):

    @article{cotton2026inversion,
    author = {Cotton, Peter},
    title = {Scalable Inversion of Contests with Correlated Performances,
             Including Softmax and Multinomial Probit},
    year = {2026},
    eprint = {2609.01133},
    archivePrefix = {arXiv},
    primaryClass = {stat.ME},
    doi = {10.2139/ssrn.7307363},
    note = {arXiv:2609.01133; also SSRN working paper 7307363},
    URL = {https://arxiv.org/abs/2609.01133}
    }

For the original independent lattice transform (`winning.classic`):

    @article{cotton2021inferring,
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
