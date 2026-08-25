# Prior art: covariance from rank data — synthesis

Assembled 2026-08-25 from two adversarial hunts (details with verbatim
quotes in `rank-covariance-prior-art-econometrics.md` and
`rank-covariance-prior-art-marketing.md`; the racing front is Peter's
separate workstream and is deliberately not covered here).

**The question.** Before proposing a paper that identifies a factor
covariance (V, D) of a Thurstonian race from multi-rank data — top-k
inclusion shares, win/place/show-style tiers — in a single market at
large N: has anyone done it?

**Answer: no direct hit, and two CLOSEs that bracket the open slot.**

## The two brackets

1. **Right object, wrong data and scale — Thurstonian psychometrics.**
   Maydeu-Olivares & Böckenholt (Psych. Methods, 2005) fit unrestricted
   AND factor-analytic (Lambda Lambda' + Psi^2) Gaussian utility
   covariances to ranking data. Böckenholt (BJMSP 1992) handles partial
   rankings including top-k patterns with constrained Sigma. But: tiny
   fixed menus (n = 4-7), individual respondents each supplying a full
   or partial ranking PATTERN, no aggregate shares, no prices, no
   algorithmic concern with N. Cite Maydeu-Olivares & Böckenholt 2005,
   Tsai (Psychometrika 2000, the gauge equivalence classes), and Dansie
   1986 (earliest gauge count) and concede this strand entirely.

2. **Right setting, wrong object — Conlon, Mortimer & Sarkis (draft,
   May 2026).** Aggregate first-choice shares plus an aggregate
   second-choice matrix, single market, no characteristics, complexity
   controlled by the rank of the second-choice matrix. But the free
   object is an NMF-style type mixture, not a utility covariance, and
   the (first, second) PAIRING must be observed. No utilities, no
   removal counterfactuals, no extrapolation across ranks. The draft's
   existence is both validation of the setting and a reason to move.

## The surviving claim, precisely

Identification and estimation of a free low-rank-plus-diagonal Gaussian
utility covariance from **aggregate marginal top-k inclusion shares**
(no observed pairings, no ranking patterns) on **large menus**, with a
fast exact forward map and inverse. Every existing estimator consumes
either full/partial ranking patterns (psychometrics, rank-ordered
probit) or an observed second-choice pairing (Conlon et al.); none
consumes marginal tier shares, none runs at N in the thousands, and
none inverts.

## The warning that gates the paper

Maydeu-Olivares (1999/2001), verbatim from the hunt: unrestricted
Thurstonian models "are not identified from univariate information
only. However, any Thurstonian model can be identified as soon as
bivariate information is employed." Marginal top-k inclusion shares are
univariate-type functionals of the ranking distribution. Whether they
identify even a rank-1 (V, D) is therefore genuinely open — the
dimension count (k tiers give ~k(N-1) numbers) is necessary, not
sufficient. Zhao & Xia 2019 give the Luce-family negative (mixtures of
Plackett-Luce barely identified from first+second choices).

**Consequence: the identification numerics experiment is the gatekeeper
for the whole program.** Before any drafting: fix N = 24 and N = 1000,
rank-1 and rank-2 truths, compute exact top-k inclusion shares for
k = 1, 2, 3, and test numerically whether (mu, V, D) is recoverable
(local Jacobian rank of the share-stack in the model parameters, then
global refits from random starts). If marginal tiers underidentify,
the fallback objects are bivariate tier moments (P(i and j both in
top k)), which the same shared-field machinery prices.

## Also useful

- Hausman & Ruud 1987: precedent that deeper ranks buy second-moment
  identification (rank-depth scale parameters in the logit family) —
  right instinct, wrong object; cite as lineage.
- Rank-ordered probit (Hajivassiliou-Ruud 1994; Layton-Levine 2003;
  Nair-Bhat 2019 MACML): general differenced Sigma but individual data
  with covariates; never covariate-free single-market identification.
- Elrod-Keane 1995 (verbatim): "the ability to infer consumer
  heterogeneity from aggregate data is limited" — the sentence the new
  paper would push against, from the founders of the factor-probit
  strand.
