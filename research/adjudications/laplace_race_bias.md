# Laplace race win-probability is biased (root cause of the winner-
# update defects reported by the bandits audit, 2026-09-04)

## Symptom (bandits session, reproduced here)
update_winner base="laplace" understates the winner's posterior
variance ~57% and overstates its mean; losers fine; normal/logistic
controls fine.

## Root cause: it is NOT the moment update or its derivatives
The lattice WIN PROBABILITY itself is biased for laplace, and the
bias is grid-converged (does not vanish as points -> inf):

  m=[0.4,0,-0.3], v=[0.6,0.4,0.5], beta2=1
  race_probabilities laplace  pts 257  [0.4965 0.2986 0.2049]
                              pts 4097 [0.4966 0.2986 0.2048]
  laplace MC (4M)             [0.4776 0.3039 0.2184]
  normal  lattice vs MC       0.4675 vs 0.4677   (exact)
  logistic lattice vs MC      0.4763 vs 0.4714   (small +0.005 bias)

The favorite's win prob is inflated +0.019 (~4%), underdogs
deflated. The engine score d logP/dm_win = 0.60 (grid-converged) vs
MC 0.55: the fixed-grid-exact gradient is the EXACT derivative of a
BIASED probability. So every downstream moment (mean over-credits
the winner, variance over-shrinks) inherits the bias, and no
second-derivative guard (_fd_eps, _clamp_d2) can cure a wrong
first-order probability. This explains why the kink fix (which cured
the variance-blowup in the order paths) did not touch update_winner's
winner coordinate.

## What is NOT the cause (ruled out)
- Base function _laplace: verified correct (unit-variance b=1/sqrt2,
  S the survival, f density, fp derivative).
- Span: laplace (18,18); tail mass at z=18 is ~4e-12, no truncation.
- Grid resolution: bias is flat from 257 to 4097 points.

## Likely mechanism and candidate fix
The laplace density has a KINK at z=0 (its own location x=m_i on the
max-lattice). forward_grid / the product-of-survivals integrand is
piecewise-smooth with a corner at each x=m_i; a lattice that does not
place a node exactly at each player's kink integrates a kinked
integrand with an O(1) systematic offset that converges to a biased
limit rather than averaging out. Logistic is smooth, so its residual
+0.005 is a separate, smaller effect (worth a look but not this bug).
CANDIDATE FIX: for kinked bases (laplace, and exponential_power with
beta<2), force the forward grid to include the per-player kink
abscissae x=m_i (and refine locally around them), then re-validate
race_probabilities against MC across configs. This is a numerical-
methods change to the core lattice, NOT a ratings patch; it must be
gated on the bandits audit harness
(bandits/tests/audit_ratings_bulletproof.py) and a race-layer MC test.
Do not patch by widening eps or clamping -- those act downstream of
the biased probability.

## Status: diagnosed, not fixed. Deliberate fix required.
