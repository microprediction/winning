# The incumbent is always multinomial logit
(Peter, 2026-09-03. The governing principle for the whole
applications program. Adopt it; re-read the adjudications through it.)

## The claim
For any problem of the form "probability over N alternatives" --
which wins, which is chosen, which fails first, which is best --
the default tool practitioners reach for is a LOGIT: softmax,
Bradley-Terry, Elo, Rasch, conditional logit, or the independence
formula 1-(1-p)^k. Different names, one model, one sin: the Gumbel /
IIA / independence assumption.

## Why it is always logit (the paper's own thesis)
Logit is the ONLY member of the discriminal-race family that is
analytically tractable at scale. Yellott 1977: softmax is exactly
the Gumbel special case of the Thurstone race. So practitioners did
not choose logit because reality is Gumbel; they chose it because it
was the only feasible choice at scale. "The incumbent is always
logit" is a corollary of "logit was the only feasible choice at
scale until now" -- which is precisely what this package removes.

## The consequence for adjudication (the correction to my error)
I kept benchmarking applications against the domain's SPECIALIZED
tool -- GHK for probit, Ford's burst-Markov for durability,
additional-greedy coverage for tests -- and finding "strong
incumbents." That is the wrong baseline. The specialized tools are
the exception; the DEFAULT the median practitioner uses is logit /
softmax / independence. The right question for any application is
not "can we beat the domain's best engine" but:
  1. Is the working incumbent a logit (independence / IIA)? Almost
     always yes.
  2. Does IIA cost something here -- are the alternatives correlated?
  3. Is the correlation STRUCTURED (factor / block / tree)?
If yes/yes/yes, the pitch is uniform and the same sentence every
time: correlated probit at the scale where you are currently forced
to use logit.

## The genuine exceptions (few, and worth naming honestly)
- Purely COMBINATORIAL incumbents that are not logit at all:
  additional-greedy coverage (set cover) in test selection. There
  the observable structure already carries the correlation and logit
  is not the foil. Rare.
- DATA problems, not method problems: Backblaze failed on
  administrative failure dates, not on the incumbent. Logit-framing
  does not rescue a data confound.
Everywhere else the foil is logit, and the market is every place
logit is used as a scale-forced approximation to a correlated race.

## Restated for positioning
The LinkedIn line "probit finally competes with logit at modern
scale" is not a tagline; it is the entire thesis and the entire
market. Every application is one instance of it. Stop apologizing
per-domain; lead with the universal foil.

## Refinement (Peter): two incumbents, split by covariance structure
The incumbent is logit ONLY when the covariance is dense/unstructured
(or assumed away). When it is SPARSE-PRECISION -- tridiagonal (AR(1),
a Markov chain), a local kernel, a tree/nested graph -- the incumbent
is a BAYES NET / Gaussian Markov random field / Kalman filter /
belief propagation. Two regimes, two foils:

  covariance is...        incumbent           winning adds
  dense / IIA assumed     BASIC logit (MNL,   the CORRELATION (basic
                          softmax, Bradley-   MNL has IIA; but see
                          Terry, 1-(1-p)^k)   below -- the FAMILY has
                                              correlation)
  sparse precision        Bayes net / GMRF /  the ORDER STATISTIC
  (tridiag, kernel,       Kalman / belief     -- P(this node is the
  tree, Markov)           propagation         max/argmin/first) --
                                              which marginal
                                              inference does NOT give
  LOW-RANK (factor)       NEITHER handles it  everything: logit
                          well                ignores the factor,
                                              GMRF can't cheaply
                                              represent low-rank
                                              (dense precision)

## CORRECTION (Peter): logit does NOT assume correlation away
Sloppy of me. Only BASIC MNL / softmax / Bradley-Terry has IIA
(iid Gumbel -> no error correlation). The logit FAMILY models
correlation richly: NESTED logit correlates within nests (closed
form, restricted to a partition); GEV / cross-nested generalizes the
nesting; MIXED / random-parameters logit approximates ANY random-
utility model including full correlation, by SIMULATING the mixing
integral. Mixed logit is the workhorse for correlated choice and --
per posttraining-luce-vs-probit-verdict -- it WINS at finite budgets.
So the foil is not "logit has no correlation"; the foil is a family
that HAS correlation, by three routes:
  MNL / softmax        no correlation (IIA), tractable base
  nested / GEV logit   restricted-structure correlation, closed form
  mixed logit          general correlation, by SIMULATION (expensive,
                       strong at finite budgets)

## The two halves of the value proposition, corrected
- vs BASIC MNL / softmax: winning adds the correlation IIA drops.
  This is the only place "adds the correlation" is honest.
- vs the CORRELATED logit family (nested / GEV / mixed): winning
  does NOT add correlation -- they have it. Winning offers it EXACT
  and STRUCTURED at SCALE without simulation (mixed logit simulates;
  nested/GEV restricts the structure), PLUS the argmax / order-
  statistic / inversion outputs the logit family does not natively
  give. The competitor is mixed logit's simulation cost and nested
  logit's structural restriction, not an absence of correlation.
- vs BAYES NET / GMRF: winning adds the ARGMAX / order-statistic
  layer. The graphical model gives marginals and the joint, but NOT
  the probability-of-maximum, the first-failure identity, or the
  k-th order statistic -- those need a separate, usually expensive,
  computation that winning does exactly over the same structure.
  (This is why the tree/nested cavity IS message passing: on those
  covariances winning competes with the Bayes net only on the
  order-statistic layer, not on the correlation modelling.)

## The home turf, sharpened
FACTOR / low-rank-plus-diagonal is where BOTH incumbents are weak:
logit ignores the shared factor, and sparse-precision methods
(GMRF, Vecchia, Kalman) represent low-rank badly because low rank in
the covariance is DENSE in the precision. So the cleanest wins are
low-rank correlation + an order-statistic query -- exactly the
factor-race the engine is built for. On sparse-precision covariances
the pitch narrows to "the argmax layer your Bayes net does not give
you"; on dense/unstructured it is "the correlation your logit throws
away."
