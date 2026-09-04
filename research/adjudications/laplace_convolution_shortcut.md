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

## RESOLVED: option (a) implemented (2026-09-04, Peter's call)
winning.ratings.nway._predictive_curves builds each player's TRUE
predictive marginal N(0, v_j) convolved with the base noise (grid
convolution against analytic Gaussian-derivative kernels, so the
density's first two derivatives carry no numerical differencing), and
every non-normal moment update prices it:

  update_winner            analytic gradient AND curvature via
                           _winner_moments_curves (no FD at all)
  update_ranking / _exact  curve rows through _order_pass
  update_winner/order_
    correlated             curve nodes (curves shared across factor
                           nodes; the factor only shifts means)
  update_order_full        _order_kernel builds curves from the
                           belief-split's psi + beta2

The normal branch is untouched (bit-stable). Prekopa keeps
log-concavity under Gaussian convolution, so _clamp_d2 remains valid;
the convolution also smooths the laplace kink, making _fd_eps's wide
differencing step unnecessary on curve paths (dropped there).

Measured (research/adjudications/predictive_referee.py, 4M x 6 MC of
the generative model, three configs including two the bandits harness
did not pick; pass mark 0.003):

  update_winner laplace   dmean 0.047 -> 0.0001   dvar 0.277 -> 0.0004
  logistic / gumbel / student4 all <= 0.001 on means and variances
  update_ranking_exact    all bases <= 0.0007 dmean, <= 0.0007 dvar
  correlated (V != 0)     laplace/logistic <= 0.0013
  winner p (laplace, config A): 0.4776 = generative MC 0.4776
  winner posterior var:   0.4910 vs MC 0.4909 (was 0.2133, the 57%)

Regression: 46 existing ratings/base tests pass; normal rows at their
historical accuracy. Pinned in tests/test_predictive_curves.py
(Gaussian-closure consistency + MC-referenced laplace winner/order
values). Remaining gate: bandits audit_ratings_bulletproof P6/P8
re-run on their side.

## Gate result (bandits audit rerun, 2026-09-04): PASSED
All P8 marks met (laplace dvar 0.2772 -> 0.0022, logistic 0.0103 ->
0.0023, gumbel 0.0032 -> 0.0006, normal unchanged); all three
non-normal bases moved, ruling out special-casing. P6 clean on every
base incl. student_base(4.0) frac|z|>10 0.073 -> 0.0000 (the
unpowered lead cleared untargeted). P7 exact order paths agree to
0.0000 (student4 2e-4). A FRESH referee on configs disjoint from
anything this side had seen (near-ties, unequal variances, belief
variance 3.0, 8-runner field): worst deviation per base <= 0.0029 --
laplace statistically indistinguishable from normal. Committed at
bandits/tests/audit_fresh_configs.py.

Residual, adjudicated NOT a defect: update_ranking (the documented
stagewise moment approximation) is overconfident, monotonically worse
with tail weight (coverage 0.890 normal ... 0.830 student4). The
normal row -- exact marginals by closure -- pins it on the
fresh-noise-per-stage decomposition, not the pricing; recorded in the
update_ranking docstring so it is not re-read as a bug.
update_ranking_exact measures 0.94+ coverage on every base.

## Refinement of the update_ranking adjudication (bandits, same day)
Two defects, not one, in the stagewise path (approx vs exact on
identical evidence, 12 seeds x 60): (1) over-shrinkage from the
fresh-noise decomposition is roughly CONSTANT, 21-28% on every
non-IIA base, and ~0% under gumbel (var ratio 0.977, mean err ratio
1.002) where IIA makes stagewise exact -- the control that proves the
double-counting mechanism; (2) the tail-weight coverage ladder is
driven by the MEAN degrading (RMS err ratio 1.20 normal -> 1.25
student4), not by extra shrinkage (which is flat to mildly LESS
severe for heavy tails). Docstring amended accordingly.

## Open lead (unchased): failure_base stagewise beats exact on mean
failure_base is the most over-shrunk (var ratio 0.718) yet its
stagewise MEAN beats the exact path (err ratio 0.689; coverage reads
0.990 because a small interval sits on a better mean). An
approximation beating the method it approximates usually indicates
something off in the EXACT path for bimodal bases -- plausibly the
FD-of-adjoint curvature or the moment projection under a two-lump
posterior. Worth a look if failure_base rankings matter; not part of
the convolution fix's gate.
