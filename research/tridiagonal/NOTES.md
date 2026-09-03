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
