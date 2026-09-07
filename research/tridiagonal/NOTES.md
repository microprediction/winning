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

## Reconciliation with "Go Forth!" (Cotton & Boofhead, JOTA-format
## working paper, github.com/microprediction/home/workingpapers)
Read in full (29 pp). What it is: THREE-SHOT OPTIMAL SEARCH on an
exponentiated OU path f(t) = exp(X_t), X_0 = b ~ N(0,1). Main
theorem: the optimal third sample is NEVER inside the interval of
the first two (except the measure-zero tie X_{t1} = b) -- "go
forth," leave the comfort zone. Two-shot solution in closed form
(flee to infinity if b<0; t* = -(1/kappa) log b if 0<b<1; stay if
b>1). Proof machinery: RAPIDITY theta = tanh^-1(e^-kappa dt) makes
OU bridge correlations compose like relativistic velocity addition
(bridge mean b tanh(theta2+theta)); an outside point at rapidity
theta2+theta has the same mean and no less variance (cosh convexity
+ Jensen), so inside is dominated. Also a human diagnostic: only
23/152 surveyed chose correctly.

RELATION TO THIS TRACK -- the dual, not the same result:
- Go Forth answers WHERE TO SAMPLE NEXT (optimal search policy on
  the path); exp2/exp3 answer WHERE THE MAX IS (the argmax law of
  the path). Same object -- extremes of a Gauss-Markov path --
  opposite direction of inference.
- The two RHYME and the rhyme is real: exp2's static law says the
  max of a positively-correlated stationary path piles up at the
  BOUNDARY (ends 2.6x at phi=0.9); Go Forth's dynamic theorem says
  never search INSIDE your bracket. The endpoint pile-up is the
  static shadow of the no-backtracking policy: mass at the boundary
  is why the inside is dominated.
- The paper's utility E[f(t_2)] is chosen to admit closed forms and
  the paper says k>3 shots gives "lengthy closed form expressions."
  The transfer operator prices what has NO closed form: P(beat the
  best seen), quantiles of best-found, expected max over the path,
  k-shot value functions by DP on the lattice -- i.e., the
  computational continuation of Go Forth beyond three shots and
  beyond mean-utility. That connects directly to research/design
  rollout pruning (same shape: sequential sampling of an evolving
  Gaussian object with a free boundary).
- Adopt the RAPIDITY coordinate: uniform-in-theta time spacing is
  the natural lattice discretization for OU chains (correlations
  additive), likely reduces grid error in exp1-3.
- Eval-design echo: the paper's inside-choice "tell" is a one-bit
  behavioral test of meta-learning -- kin to the evalstats
  diagnostics program.

## Go Forth now has a home: github.com/microprediction/browniansearch
Created 2026-09-03. The paper lives there as the founding working
paper (canonical dated copy stays in microprediction/home, first
committed 2022-04-06); browniansearch/brownianbandit are the
search-one/prune-many siblings, and the k-shot lattice DP belongs in
browniansearch when built, using this track's transfer operator.

## Correction to the Go Forth reconciliation above (2026-09-03,
## later the same day)
Cleaning the paper for browniansearch surfaced an error in its
Section 5: the mean-matched outside point's variance is sech^2 of
the composed rapidity, not sech, and with the correct value the
no-backtracking theorem FAILS on a positive-measure region --
verified by exact conditioning and 2m-sample weighted MC
(browniansearch/verify/check_inside.py). The true result, now in the
rewritten paper (Go Forth on Bad News, browniansearch/papers/
go_forth): interior bridge points DOMINATE their mean-matched
outside rivals (variance ratio cosh(th2+th)/cosh(th2-th) >= 1), the
bridge middle beats every outside choice iff b^2(1+rho)-4b sqrt(rho)
+2rho < 0 (opens at b*=0.4233 along rho=b), and the corrected maxim
is: go forth on bad news, revisit the bridge on good news. The rhyme
with exp2 stated above should be read accordingly -- the boundary
pile-up of the argmax still explains why OUTSIDE is right on bad
news, and the bridge's retained variance explains why INSIDE is
right on good news. The rapidity-spacing suggestion for the OU
lattice stands.

## Correction (external review of the grass paper, 2026-09-03): the
## rapidity-spacing suggestion above is WRONG
Correlations across consecutive OU intervals MULTIPLY,
rho(t+s) = rho(t) rho(s), so the additive coordinate for interval
composition is kappa t = -log rho -- NOT the rapidity. The
tanh-addition identity is an equal-anchor bridge-MEAN-matching
device, not a composition law. Uniform-in-theta lattice spacing has
no additivity property; disregard the two "adopt rapidity spacing"
remarks above. (Also corrected in the paper: the variance
amplification of interior over mean-matched exterior points is the
CONSTANT (1+rho)/(1-rho) = e^{2 Theta} across the whole bracket, and
staying at an observed anchor is optimal above b = e^{2 Theta} --
the grass claim is conditional, not universal.)

## exp4_bandlift: the band-diagonal lift, measured (2026-09-05)
The correction above is now realized: bandwidth 2 (stationary
unit-variance AR(2), Yule-Walker sig^2 and rho1) via the lifted state
Z_t = (X_t, X_{t-1}), one einsum per step against an (a, c, b)
kernel, O(n L^{b+1}).
- Embedding: phi = (0.9, 0) on the SAME L=200 grid reproduces exp1's
  bandwidth-1 pass to 5.6e-16 -- the lift is exact, not approximate.
- Genuine bandwidth-2 vs 400k MC (n=50, L=200, grid-limited like
  exp1): smooth (0.5, 0.3) max err 0.011; persistent (0.6, 0.35)
  0.012; CYCLICAL (1.6, -0.8) 0.011 -- complex roots, oscillatory
  autocorrelation, the regime with no bandwidth-1 analogue at all.
  ~230 ms per threshold.
- Linear in n confirmed on the lifted pass: n=50 -> 200 scales x3.1
  (kernel precompute amortizes the gap to the ideal x4).
Open from this file's earlier promises: the GHK-degradation
head-to-head on AR(1) (Ridgway's exponential-variance regime), the
argmax forward-backward on the lifted state (exp2's law for AR(p)),
and quadrature/low-rank message compression past b ~ 3.

## The argmax layer shipped as a package (2026-09-06)
julia/GMRFExtremes generalizes exp1-exp3 off stationary AR(1) to any
Gauss-Markov chain (non-stationary mu/phi/s, or a tridiagonal
precision via its bidiagonal Cholesky) and ships argmax marginals,
max-CDF/expected max, first passage and excursion probabilities with
a GaussianMarkovRandomFields.jl dispatch extension -- the
contribution named in planning/julia_bayesnet_contribution.md. Its
tests re-derive the exp2 boundary pile-up (ratio gate 2.0-3.4 at
phi = 0.9) and the exp3 discrete arcsine U-shape as living
regressions. The exp4 lifted-state argmax (bandwidth 2) remains the
open item, now with a named home when built.

## exp5_sp500_argmax: the drifted argmax law on 98 years of real data
## (2026-09-06)
The day of the market's yearly high, ^GSPC daily closes 1927-2026
(Yahoo chart API, fetched at run time), 98 full years. Empirical
decile law of the within-year argmax is the arcsine U-shape tilted
hard to year-end: deciles (1, 10) carry (0.163, 0.408). Engine
(GMRFExtremes argmax_marginals on the standardized 63-block drifted
walk, delta = 2 mu/sigma = 0.041/step): drifted law log-likelihood
-191.1 over the 98 years vs closed-form driftless arcsine -198.4 and
uniform -225.7 -- the drift correction, which has no closed form, is
worth 7.3 nats. Honest residual: the empirical year-end pile-up
(0.408) exceeds the drifted-Gaussian prediction (0.269); the gap
measures what the homoskedastic Gaussian walk misses (vol
clustering, momentum). Methodological note that cost an hour: at
daily resolution n = 252 the argmax grid cannot afford to resolve
the step scale and near-tie mass leaks (0.907 total); the 4-day
block aggregation is EXACT for the argmax's block and makes the law
scale-free. MISFILED, corrected 2026-09-06: this was written as the win-nodes
paper's example and does not belong there. It contains no graphical
model, no observations and no posterior -- it is an argmax law of a
PRIOR, which is brownian-max material for the browniansearch line
(see the Go Forth reconciliation above), not Bayes-net material. The
paper's example is exp6. Keep this as what it is: a validation of the
drifted argmax law against real data.

## exp6_career_peak: the win-nodes paper's actual example (2026-09-06)
A Bayes net whose evidence is wins. Latent yearly ability path per
player with a random-walk prior; each ATP match contributes
log Phi(+-(theta_t - a_opp)), one year per term, so the likelihood
Hessian is diagonal, the prior precision tridiagonal, and the Laplace
posterior precision is TRIDIAGONAL -- a Gauss-Markov chain, which is
exactly chain_from_precision's input. 129,559 matches 1985-2024,
1,453 players (static field abilities by MAP probit Bradley-Terry;
opponents then held fixed, a conditioning step not a joint fit); tau
by Laplace marginal likelihood.

Fitted peaks land on the historically argued seasons (Federer 2006,
Djokovic 2015, Sampras 1994, Agassi 1995, Murray 2016), so the point
answers are credible and the uncertainty is the finding:
  Federer  2006 P 0.52 (2005 carries 0.41)
  Nadal    2013 P 0.267 -- BIMODAL with 2018 at 0.220
  Agassi   1995 P 0.551, 80% credible set {1995, 2002, 2003}, NOT an
           interval: the 2002-03 resurgence is a genuine rival peak
  Djokovic 0.428, Sampras 0.465, Murray 0.593
MC referee 200k posterior draws per career: max |exact - MC| 0.0019.

The mechanism, visible in the same fit: at equal posterior mean the
SPARSER year carries more argmax mass (Nadal 2010, mean 2.042, 80
matches, P 0.066 vs 2012, mean 2.051, 48 matches, P 0.088). Fewer
matches leave a wider posterior and a wider posterior has more room
to hold the maximum. No error-barred rating table contains that
comparison.

Package bug this experiment caught: chain_from_precision returned the
chain in REVERSED index order, and the existing test could not detect
it because a stationary chain's argmax law is symmetric. Fixed by
factorizing the reversed problem; a drifted-chain test now pins the
orientation.
