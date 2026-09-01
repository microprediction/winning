# Completing covariance matrices (Peter, 2026-09-01)

Races only reveal covariance information about contestants who raced
TOGETHER: the observation pattern of Sigma is a union of cliques, one
clique per historical field. Completing the rest is the classic
positive-semidefinite matrix completion problem, and the winning
machinery touches it from four sides.

## 1. The classical theory maps on exactly
Fields are cliques of the observation graph. PSD completion of a
partial matrix with a CHORDAL pattern exists iff every fully observed
principal submatrix is PSD, and the maximum-determinant completion is
the canonical unique choice -- the one whose INVERSE is zero on every
unobserved entry (covariance selection). [U: Grone-Johnson-Sa-Wolkowicz
1984 for the chordal theorem; Dempster 1972 for covariance selection;
Vandenberghe-Andersen for the max-det computation.] The max-det
objective is the same logdet machinery the D-optimal scheduler just
built.

## 2. Tournament design FOR completability
The scheduler chooses which cliques get observed, so "design the
season so the covariance is completable" = build a chordal clique
cover of the population with race-sized cliques -- a combinatorial
design criterion that composes with (or competes against) the
D-optimal information criterion. A season whose fields never overlap
enough leaves Sigma unidentifiable no matter how many races run;
chordality is the checkable certificate. Candidate result: the
marginal value of a field = information gain PLUS the completability
it unlocks.

## 3. Grammar completion (implementable now)
fit_covariance already fits factor + blocks + diagonal to a DENSE
Sigma by the projected objective; the completion variant masks the
residual to observed entries:

    min_{V, D}  || M o P(Sigma_obs - VV' - diag D)P ||_F^2

with M the clique mask -- same alternation, masked V-step (an
eigenproblem becomes a masked low-rank fit; alternate over columns) and
the D-step unchanged on observed diagonals. The fitted grammar then
PRICES the completion, and cross-clique entries come out as the
grammar's implication rather than a free completion -- a structured
alternative to max-det, and the two disagree exactly where the grammar
is informative. Measuring that disagreement against a held-out truth
is the natural first experiment.

## 4. The choice-relevant twist (the novel question)
The engine never needs Sigma, only P Sigma P -- and race OUTCOMES only
ever identify choice-relevant functionals of the within-field blocks
(difference variances; tie densities pin them at the boundary). So the
completion that matters is completion of the QUOTIENT, a weaker
requirement than PSD completion of Sigma itself: completions of Sigma
that disagree only in the choice-irrelevant directions of each field
are observationally equivalent. Characterizing that equivalence class
-- what does a full season of race data determine about Sigma, and
what never? -- looks like the paper-grade question, with Mosteller's
common-correlation invariance as the k=1 seed of the story.
