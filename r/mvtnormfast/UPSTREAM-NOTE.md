# Draft note to Torsten Hothorn (mvtnorm maintainer)

Subject: A fast exact path for pmvnorm on factor-structured covariance

Dear Torsten,

mvtnorm has been the reference for multivariate normal probabilities in
R for two decades, and a project of mine leans on it daily -- thank you
for maintaining it.

I wanted to share a computational observation that may interest you.
When the covariance has factor-plus-diagonal structure, sigma = VV' +
diag(D), rectangle probabilities reduce to a low-dimensional smooth
integral of products of univariate normal CDFs (conditional on the
factor, coordinates are independent). Evaluated on adaptive
Gauss-Hermite or low-discrepancy nodes this is deterministic and very
fast: on my machine, cases where pmvnorm needs 31 ms (n=30) to several
seconds (n=200, tight tolerance) evaluate in one or two milliseconds,
agreeing with pmvnorm to within its own reported error bound, and with
Botev's minimax tilting in the deep tails.

I have packaged this as a small companion, mvtnormfast
(github.com/microprediction/winning, r/mvtnormfast; a Python sibling
exists for scipy). It is deliberately narrow: it detects exact
factor-plus-diagonal structure (strict verification) and falls back to
pmvnorm unchanged otherwise, so it accelerates a slice of pmvnorm's
traffic rather than replacing anything.

If a structured fast path inside mvtnorm itself would ever be welcome,
I would be glad to contribute it, benchmarks and tests included. And if
not, no matter -- I mainly wanted the observation and the numbers to
reach you.

With best regards,
Peter Cotton

---
STATUS: sent by Peter to Torsten.Hothorn@R-project.org on 2026-08-28.
