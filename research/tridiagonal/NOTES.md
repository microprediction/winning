# Tridiagonal / Gauss-Markov order statistics (research track)
(Opened 2026-09-03 at Peter's direction. The sparse-precision regime,
where the incumbent is a Bayes net / Kalman / GMRF -- NOT a logit.
See [[the-incumbent]] and, emphatically, [[never-strawman-iid-logit]].)

## The object
A tridiagonal precision matrix is a Gauss-Markov chain: a
one-dependence process (AR(1) stationary, or a time-varying
Gauss-Markov / state-space model). Covariance is dense but precision
is banded. The order-statistic queries:
- max CDF  P(max_t X_t <= x)  and the running maximum,
- argmax   P(X_t = max_s X_s) for every t (which index is the peak),
- first passage  first t with X_t > barrier (= first-failure over a
  temporally-correlated race),
- k-th order statistic / how many cross a level.

## The incumbent here is a graphical model, and it is honest
For a tridiagonal/Markov covariance the practitioner uses a Kalman
filter / GMRF / belief propagation. That machinery HAS the
correlation and computes marginals and the joint. What it does NOT
give is the argmax / order-statistic / first-passage distribution --
P(X_t is the max) is not a Kalman output. That is the layer this
engine adds, computed EXACTLY by a forward-backward pass on the chain
(which IS the Kalman smoother / sum-product). The pitch is not "we
model the correlation" (the Bayes net does) -- it is "we compute the
argmax/order-statistic over your Gauss-Markov model, which your
filter does not." NOT a logit anywhere in this track.

## Why tridiagonal specifically, and where it beats GHK
Simulation degrades badly here: Ridgway (arXiv:1411.1314) showed GHK
on an AR(1) Gaussian is sequential importance sampling whose
normalized variance grows EXPONENTIALLY in dimension. So the
tridiagonal orthant/max probability is exactly the case where the
simulation incumbent (GHK) is worst and an exact transfer-operator
pass is best -- a clean head-to-head. The exact method is a 1-D
forward integral (transfer operator restricted below the threshold),
O(n L) per threshold, linear in n.

## Applications (correlated first-passage / extremes over time)
- Time-series extremes: max / running-max / max-drawdown of an AR(1)
  or Gauss-Markov series; first barrier crossing (finance: first time
  a spread/price crosses a level; the min/first-passage race).
- Streaming / sequential monitoring: first time a temporally
  correlated signal breaches an SLO -- first-failure over a Markov
  race (connects to research/evalstats stopping and research/design
  first-passage boundaries).
- Reliability with WEAR (aging = temporal correlation): degradation
  as a Gauss-Markov process, first-passage to failure -- the honest
  version of the Backblaze idea, where the correlation is in TIME and
  modelable, not a reporting artifact.
- Spatial 1-D / nearest-neighbor kernels (a line of sensors): which
  location is the extreme under a banded kernel.

## First probe (exp1_maxcdf)
Exact max CDF and expected max of a stationary AR(1) Gaussian via the
forward transfer operator restricted below the threshold, validated
against Monte Carlo, timed to confirm linear-in-n. Next: the argmax
vector by forward-backward (the Kalman-smoother-plus-order-statistic
the filter does not give), and the GHK-degradation head-to-head on
AR(1).

## exp1_maxcdf measured (2026-09-03)
Exact AR(1) max-CDF via the forward transfer operator vs 400k-draw
Monte Carlo, n=50:
  phi=0.0:  max|err| 0.014 (grid-limited at L=400)
  phi=0.5:  0.013
  phi=0.9:  0.006
  phi=0.99: 0.005   P(max<=1.5)=0.809 in 3 ms
Matches MC within noise, tightest at HIGH correlation -- exactly
where GHK's variance blows up (Ridgway). Linear in n confirmed:
3 ms (n=50) -> 12 ms (n=400). The exact order statistic on a
Gauss-Markov chain, cheap, on the covariance the simulation incumbent
handles worst. Next: the argmax vector by forward-backward (the layer
the Kalman filter does not give), the first-passage distribution, and
a timed GHK head-to-head on AR(1).

## exp2_argmax: P(X_t is the max), the layer Kalman lacks (2026-09-03)
Forward-backward argmax over AR(1), vs 400k MC, n=20 (max err <0.007,
ratios match; sum 0.90-0.95 is grid mass loss at L=240, tighten to
normalize):
  phi=0.0: uniform 0.05 (recovered) -- iid has no preferred peak.
  phi=0.5: ends 0.056 vs middle 0.046.
  phi=0.9: ends 0.091 vs middle 0.035 -- ENDPOINTS ~2.6x likelier to
    be the max.
The non-obvious, correct finding: positive AR(1) correlation pushes
the maximum to the BOUNDARY. Interior points are squeezed by
correlated-high neighbors on both sides; endpoints have one neighbor.
A STATIONARY KALMAN FILTER GIVES IDENTICAL MARGINALS EVERYWHERE and
cannot express this -- it is precisely the order-statistic the engine
adds over the graphical model. Clean demonstration of the track's
value proposition.

## Schur-damped block races (Peter, 2026-09-03): another grammar
schur.microprediction.org: Schur-complement DAMPING interpolates a
block covariance between isolated blocks (gamma=0) and fully merged
(gamma=1) via A - gamma B D^-1 B' (the damped sub-covariance,
Cotton's Schur-damping / HRP-to-min-variance bridge). A SCHUR-DAMPED
RACE prices the argmax / order statistic under that partially-merged
block covariance -- a one-parameter family of correlated races from
independent-blocks to fully-coupled. Natural grammar extension: the
block kernel with a damping gamma on cross-block coupling, and the
engine's shared-field prices the winner distribution along the whole
gamma-bridge. Connects the winning race engine to Peter's own
allocation/Schur work (same author, adjacent package): the Schur
covariance FEEDS the race. Candidate probe: argmax/PoM of a two-block
Schur-damped race swept over gamma, exact vs MC, showing how the
win-probability vector moves as the blocks merge. Filed here; belongs
with the covariance-grammar thinking ([[the-incumbent]]), not
tridiagonal per se.

## exp3_arcsine: the engine recovers Levy's arcsine law (2026-09-03)
Peter noted exp2 resembles his work on the max of a Brownian path.
It does, exactly: the argmax of a RANDOM WALK (phi=1, integrated,
Markov) has the arcsine time-of-maximum law (Sparre Andersen; Levy
continuum 1/(pi sqrt(tau(1-tau)))). Ran the transfer-operator argmax
on a random walk:
  n=20: max|engine-MC| 0.003, max|engine-arcsine| 0.012
  n=40: max|engine-MC| 0.004, max|engine-arcsine| 0.008
The engine matches the FINITE-n walk (MC) to 0.3%% and the CONTINUUM
arcsine to ~1%% -- correct, because it computes the exact finite-n
law and arcsine is its n->inf limit, so it tracks the finite walk
tighter than the continuum does. Endpoint pile-up (n=40: ends 0.093
vs middle 0.016, ~6x) is the discrete arcsine U-shape.

IMPLICATION / the contribution vs the classical law: arcsine is a
CLOSED FORM only for driftless, barrier-free, unit-increment,
n->inf Brownian motion. The transfer operator gives the exact
argmax (and max value, first-passage) for the cases with NO closed
form: finite n, DRIFT (asymmetric increments), reflecting/absorbing
BARRIERS, time-varying or non-Gaussian increments, and general
Gauss-Markov (mean-reverting) paths. That is a computational engine
for the time-of-maximum of a correlated path where the arcsine
special case does not apply -- and it connects the winning race
engine to Peter's max-of-Brownian-path paper (whose specific method/
scope to reconcile with this transfer-operator approach).

## Correction (Peter, 2026-09-03): the track is BAND-DIAGONAL,
## tridiagonal is just bandwidth 1
Peter: "I think I really meant band diagonal." Right generalization.
A precision matrix of bandwidth b means each point depends on its b
nearest neighbors -- a b-dependent process. The transfer-operator
trick survives via STATE LIFTING: the vector (X_t, ..., X_{t+b-1})
IS Markov, so the same forward(-backward) pass runs on the lifted
state. Cost O(n L^b): still linear in n, exponential only in the
bandwidth -- fine for b = 2, 3 (AR(p), local kernels), and my exp2
ELI5 overstated this as a hard wall. Beyond small b, use quadrature/
low-rank compression of the lifted messages rather than a full grid.
Incumbent unchanged (banded GMRF / Kalman with state dimension b,
which has the correlation and not the order statistic). exp1-exp3
are the b=1 case; the AR(p) lift is the natural exp4.
