# The laplace rating-update bias is the convolution shortcut, not the
# lattice (corrected diagnosis, 2026-09-04)

## Correction to an earlier note
An earlier version of this file blamed forward_grid's handling of the
laplace kink. That was WRONG and is retracted. The lattice is
correct. Credit to the bandits session for the correct diagnosis.

## Verified root cause
race_probabilities(a, D=., base=B) computes a race among PURE base-B
performances of variance D_i. Verified against MC to 2e-4:

  m=[0.4,0,-0.3], v=[0.6,0.4,0.5], D=v+1, base=laplace
  lattice                    [0.4966 0.2986 0.2048]
  MC pure-laplace var=D      [0.4964 0.2985 0.2051]   maxerr 0.0002
  MC N(m,v)+unit laplace      [0.4773 0.3040 0.2187]   maxerr 0.0193

The moment updates form the predictive dispersion as D = v + beta2
(nway.py:103 update_winner; :321 update_ranking_exact; :427
_mixture_update; :128 the pairwise pair-variance) and hand it to the
base. Folding a Gaussian BELIEF variance v into the base dispersion
is exact ONLY for base="normal", because only the Gaussian is stable
under convolution. For laplace/logistic/gumbel the true predictive is
N(m,v) convolved with the base noise, which is NOT a rescaled base;
the shortcut replaces it with a pure, wider base, and the error grows
with distance from Gaussian (normal 1e-4, gumbel 2e-3, logistic 5e-3,
laplace 1.9e-2). This is why update_winner laplace overstates the
winner mean and understates its variance ~57%, and why the exact
order paths are overconfident: the derivatives are exact for the
wrong predictive.

## Contract clarification (the low-risk fix, applied)
race_probabilities and the moment-update docstrings now state that D
is the dispersion of the base itself and that folding a belief
variance into D is valid only for the Gaussian base. No behaviour
change.

## Open decision for Peter (behavioural, NOT made here)
For uncertain abilities under a non-Gaussian base the update needs
the genuine convolution (a Gaussian-smoothed base density on the
lattice), or the non-Gaussian moment updates should be restricted /
warned. Options:
  (a) implement a convolved-base entry point (Gaussian smoothing of
      the base density, variance v, before the race) -- correct and
      general, real work;
  (b) raise/​warn when a non-normal base is combined with nonzero
      belief variance;
  (c) document only (current state) and leave callers responsible.
Gate any behavioural change on bandits/tests/audit_ratings_bulletproof.py
plus a race-layer MC test. NOTE: the winning inversion PAPER and its
Gaussian factor path are unaffected -- this is only the non-Gaussian
RATING-update use.
