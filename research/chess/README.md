# Chess ratings: beating a deployed rating system

Moved here from the `bandits` repo, where the measurement programme
lives. This is a rating-system result against a rating system actually
in production, which makes it `winning`'s subject matter rather than
bandits'. The Chatbot Arena work and the accompanying paper stay in
bandits: at two entrants the engine reduces to a probit link, so the
engine is incidental there and claiming it here would overstate its
role.

## The result

On 59,399 decisive Lichess games (2013-01, 639 players with >= 100
games), a partially pooled factor rating — one level plus shrunk
bullet and blitz offsets, with shared colour terms — beats **Lichess's
deployed architecture**, which is Glicko-2 maintained separately per
time control.

| arm | seed 0 | seed 11 | seed 22 |
|---|---|---|---|
| Glicko-2 pooled | 0.6323 | 0.6304 | 0.6365 |
| **Glicko-2 per time control** *(Lichess)* | 0.6234 | 0.6241 | 0.6294 |
| our scalar | 0.6117 | 0.6131 | 0.6145 |
| **our factor** | **0.6073** | **0.6104** | **0.6112** |

Held-out log loss; three independent splits; Glicko-2's `tau` tuned on
validation exactly as our ridge is; every arm predicts the same games
from the same information.

| comparison | seed 0 | seed 11 | seed 22 |
|---|---|---|---|
| **factor vs Lichess** | **−0.0161** | **−0.0137** | **−0.0182** |
| estimator (our scalar vs Glicko-2 pooled) | −0.0206 | −0.0173 | −0.0219 |
| structure (factor vs our scalar) | −0.0044 | −0.0027 | −0.0033 |
| stratification's value to Glicko-2 | +0.0089 | +0.0063 | +0.0071 |

Every interval excludes zero on every split, P(better) ≥ 0.995.

**Do not quote the headline without the decomposition.** Roughly four
fifths of the margin is the *estimator* — a batch Thurstonian MAP fit
against Glicko-2's online update — and one fifth is the *factor
structure*. The structure's own number, 0.003–0.004 with intervals
excluding zero on all three splits, is the claim this library's factor
API actually supports.

## Robustness

**Colour is not the confound** (`exp36`). Our arms model white
advantage; standard Glicko-2 does not, so a referee's first question
is whether the estimator column is really colour bookkeeping. It is
not: colour is worth −0.0016 of the −0.0206 gap, and a colour-blind
fit still beats colour-blind Glicko-2 by **−0.0190** like-for-like.
The honest footnote is that ~8% of the estimator column is colour.

**Open, and ranked by what a referee hits first:**

1. **Batch vs online.** Our estimator sees the training set at once and
   can revisit; Glicko-2 processes each game once because a production
   system must. Part of the estimator column is that privilege. The
   deployment-realistic comparison is online-vs-online, now runnable
   via `winning.ratings.AbilityTracker`.
2. **One month of 2013.** 639 players, early-adopter Lichess, a rating
   system that has since changed.
3. **Draws dropped** (3.3% here, far more in classical and at
   strength). The Arena work has a proper three-way tie model; this
   does not.
4. **The ≥100-game threshold** keeps the dense subpopulation, where
   rating is easiest — Lichess must rate everyone. This one carries a
   prediction: pooling should help sparse players *most*, so the
   factor margin should widen as the threshold drops. If it narrows,
   that is evidence against the mechanism.

## Why stratification helps them and not us

The two facts look contradictory and are not:

| base estimator | stratifying it |
|---|---|
| Glicko-2 pooled, 0.6323 (weak) | **gains** 0.0089 |
| our scalar, 0.6117 (strong) | **loses** 0.0006 |
| our scalar → pooled factor | **gains** 0.0044 |

Stratification's payoff runs inversely to the quality of the base
estimator: its cost is sparse-cell variance and its benefit is capped
by what pooling already captures. Partial pooling extracts the same
signal without the variance cost, which is why it wins in both
regimes — and why the shrunk-offset form is the unconditional
recommendation. A user reporting that stratification helped them has
told you about their base estimator, not found a counterexample.

## The dimension search, including its failures

The covariate was not obvious, and the first choice failed. Both
failures are kept because they are the transfer conditions:

- **`exp29`, `exp29b`, `exp29c` — opening family FAILS.** A fair fit
  (offset ridge swept to the scalar limit) collapses exactly onto the
  scalar. The axis is nonetheless real: among 790 players with >= 20
  games in each family, the style differential carries a true sd of
  ~4.7 win-rate points, ~4σ above binomial noise. It is
  **unidentifiable, not absent** — players self-select openings
  (tactical share bimodal, deciles 0.00/0.10/0.86/0.98/1.00, sd 0.415
  against a no-choice null of 0.058), so the within-player contrast
  identification needs barely exists.
- **`exp30` — time control WORKS.** 327 of 632 players have >= 20
  games in >= 2 controls. The tuned offset ridge is finite and
  interior where openings fled to the scalar limit.
- **`exp32` — the K=2 factor projection is a whisper.** Loadings
  derived from the fitted style offsets, priced through
  `race_probabilities(mu, V=...)`: the tuned scale is interior and the
  improvement sits entirely in the style-mismatched half (−0.0003),
  exactly where the mechanism puts it, but it is not established. At
  two entrants a shared factor enters the margin only through
  (v_i − v_j)², a ~2% variance perturbation. This is the honest
  statement of where correlation does *not* pay.
- **`exp31` — the giant-killer axis is registered and unrun.** Opponent
  strength passes the exogeneity check by construction.

> **Transfer conditions, both checkable before fitting.** A factor
> rating needs (a) abilities static relative to the covariate — on
> Formula 1, constructor identity loses to a driver-only baseline once
> that baseline can forget — and (b) a covariate that varies
> *exogenously*. Models do not choose their prompts, so Arena works.
> Players choose their openings, so that fails. `covariate_contrast_report`
> in `winning.ratings` runs check (b) in one line.

## Files

| | |
|---|---|
| `exp29*.py` | opening family: the failure and its two diagnostics |
| `exp30_chess_time_control.py` | the working dimension |
| `exp31_chess_giant_killers.py` | registered, unrun |
| `exp32_style_heteroskedastic.py` | the K=2 factor projection |
| `exp35_vs_glicko2.py` | the real Glicko-2 comparison |
| `results/` | per-game losses and raw outputs |

Data is one month of the Lichess open database
(`database.lichess.org`, CC0), cached to
`~/.cache/winning/lichess_2013_01_headers.parquet`; `loader.py` builds
it. Estimators import `winning.ratings.factor_ratings`; the originals
used the bandits reference implementation and were swapped on the
move, with `exp30` re-run afterwards to confirm the numbers reproduce.

**Open:** all of this is one month of 2013. A modern month has orders
of magnitude more players and a Lichess rating system that has since
evolved. That replication is the obvious next step and is not done.
