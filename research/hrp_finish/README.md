# Finishing HRP: how much, and with which matrix

Question (Peter): polish HRP with the covariance it did NOT use -- how much,
and which matrix? Formalized as transport: invert w_HRP to abilities under
HRP's own belief (the cophenetic matrix, rank-3+diag approx), re-price under
(1-gamma) coph + gamma R_hat. gamma=0 is an exact null (reproduces HRP);
gamma=1 prices under the full sample covariance.

Result (n=30, T=90, 150 trials, realized vol under the true covariance):

    EW +1.53% vs HRP | HRP baseline | gamma .25/.50/1.0: -0.13/-0.25/-0.42%
    min-var(ridged): -4.07%

Readings:
1. Direction confirmed: the unused covariance helps, monotone in gamma; at
   this sample size there is no interior optimum (Sigma_hat well-estimated).
   The Schur-allocation interior-gamma result should reappear at smaller T.
2. Magnitude: transport repairs the BELIEF while preserving HRP's CHARACTER
   (still a race-functional, diversified allocation). Most of min-var's
   -4% comes from changing the objective, not the belief. Finishing is a
   repair, not a bridge to Markowitz -- by design.
3. The "unused" ledger has three lines (cross-split blindness; within-
   cluster texture; dendrogram lossiness incl. seriation) and this blend
   lumps them. Decomposition (fixed-topology vs re-clustered) is the next
   experiment, per Peter's point that seriation itself uses R imperfectly.


## T=45 (noisy regime): shrinkage question answered

    HRP baseline | raw-fit gamma .25/.5/1.0: -0.11/-0.20/-0.32%
    LW-shrunk-then-fit gamma .25/.5/1.0: +0.01/+0.03/+0.08%  (HURTS)
    MinVar(ridged): -0.85%  (collapsed from -4.07% at T=90)

Three conclusions:
1. NO explicit shrinkage: Ledoit-Wolf toward identity deletes exactly the
   signal the finishing uses -- cross-cluster correlation -- and turns the
   improvement into a small loss at every gamma.
2. The rank-k+diagonal structural fit IS the right regularizer: even at
   T=45 the raw-fit ladder stays monotone to gamma=1 with no interior
   optimum. Structure-as-shrinkage suffices.
3. Robustness inversion: RIDGE min-var's edge collapsed 5x under noise
   (-4.07% -> -0.85%) while race-finishing held (-0.42% -> -0.32%).
   (Note: ridge min-var is a strawman; see the corrected ladder below
   with Ledoit-Wolf and structured opponents.)

FINAL RECIPE: invert w_HRP under the cophenetic belief; re-price under the
rank-k+diag fit of the RAW sample covariance; gamma = 1; no other shrinkage.


## The full T-ladder with real min-var opponents

An earlier draft of this note claimed a "crossover" where finished HRP
beats min-var at T <= n. That claim was AGAINST A STRAWMAN (ridge-only
min-var) and is retired. With properly shrunk opponents:

    vs HRP (negative = better)      T=90     T=45     T=30(=n)  T=20(<n)
    min-var (ridged, strawman)     -4.07%   -0.85%   +2.02%    +0.60%
    min-var (Ledoit-Wolf linear)   -4.35%   -2.28%   -1.43%    -0.80%
    min-var (rank-3+diag "struct") -5.32%   -3.53%   -2.61%    -1.48%
    finished HRP (gamma=1)         -0.42%   -0.32%   -0.31%    -0.28%

Corrected findings:
1. NO CROSSOVER vs competent min-var: LW and struct min-var beat plain
   and finished HRP at every T, including T < n. Only the un-shrunk
   ridge strawman collapses below T ~ 1.2n. If you are willing to
   abandon HRP's character for the min-var objective, do that.
2. NEAR-INVARIANCE stands: the finishing repair is essentially constant
   (-0.42/-0.32/-0.31/-0.28%) across a 4.5x range of estimation quality
   while every min-var arm swings by points. Shrinkage-via-the-race in
   one row: bounded transport sensitivity makes the edge independent of
   estimation noise. The honest pitch for finishing is not "beats
   min-var" but "a free, always-on repair that keeps HRP's character".
3. SIDE RESULT: the race family's own covariance estimate -- the
   rank-3+diag fit of the raw sample correlation, rescaled by vols --
   is the best min-var input tested, beating linear Ledoit-Wolf at
   every T (e.g. -2.61% vs -1.43% at T=n). Structure-as-shrinkage
   beats scalar shrinkage here. Consistent with the T=45 finding that
   LW-then-fit deletes cross-cluster signal.

CAVEAT ON 3 (Peter): likely a function of the generative model. The
synthetic truth is market beta + 3 sector factors + idio, i.e. itself
~rank-4+diag, so the rank-3+diag estimator is near-well-specified and
the comparison flatters it (an inverse crime). Fair tests, not yet run:
(a) out-of-class truths -- dense residual correlation beyond the
factors, more true factors than fitted, heavy tails; (b) the stronger
published benchmark, Ledoit-Wolf NONLINEAR shrinkage (2017); (c) real
returns, below.


## Real data: Ken French 30 industry portfolios (run_real.py)

Walk-forward on ~50 years of daily returns (no one chooses the
generative model): monthly rebalance, trailing T-day estimation window,
21-day hold, all arms long-only, realized annualized vol over ~616
out-of-sample rebalances.

    vs HRP (negative = better)      T=60     T=30(=n)
    EW                            +14.75%  +15.04%
    min-var (ridged)               -1.19%   +8.18%
    min-var (Ledoit-Wolf linear)   -4.98%   -3.01%
    min-var (rank-3+diag "struct") -7.45%   -6.77%
    finished HRP (gamma=0.5)       +0.95%   +0.61%
    finished HRP (gamma=1)         +1.45%   +0.74%

Findings, including one failed prediction:
1. THE FINISHING FLIPS SIGN ON REAL DATA. The lab's monotone-in-gamma
   improvement (-0.3..-0.4%) becomes a monotone-in-gamma HARM (+0.7 to
   +1.5%). The prediction on record -- that bounded sensitivity would
   carry the repair through -- was WRONG as to sign (right as to
   magnitude: the damage is bounded at ~1%, not a blow-up). Diagnosis:
   the lab truth was STATIONARY, so R_hat's detail beyond the tree was
   unbiased signal about the scoring covariance. In real returns the
   fine structure is largely transient; the coarse dendrogram is the
   persistent part. HRP's lossiness functions as shrinkage, and
   finishing un-shrinks it.
2. Struct's win over LW SURVIVES real data (-7.45% vs -4.98% at T=60;
   -6.77% vs -3.01% at T=n), so it is not purely a lab artifact --
   with Peter's caveat still standing, doubled: industry portfolios
   are pre-aggregated baskets, factor-like by construction of the
   assets. This reads "rank-k+diag is an excellent min-var input for
   baskets", not "markets are factor-like". Single names are the
   harsher test, and nonlinear LW remains unbenchmarked.
3. The ridge strawman's T<=n collapse IS real on real data (+8.18% at
   T=n) -- min-var's classic failure regime exists; competent
   shrinkage simply walks through it.

## "Use the corr you left out" -- denoised and slow variants
   (run_real2.py, run_real3.py)

The gamma-blend adds the FULL residual E = R_hat - coph: ~400 raw
entries. Peter's reading -- add only the structure HRP is blind to --
suggests rank-1/2 eigencomponents of E instead (a cross-branch link is
a rank-1 object; a handful of parameters aimed at the blind spot).
T=60 real data:

    full-g1 +1.45% | r1-g0.5 +0.83% | r1-g1 +1.39%
                   | r2-g0.5 +0.66% | r2-g1 +0.99%

Denoising does NOT rescue the finishing: even the single dominant
left-out eigencomponent at half strength hurts. At the fast window the
top eigenvector of E is itself dominated by transient structure.

Remaining hypothesis (run_real3.py): the left-out corr is real but
SLOW -- measure the tree from the fast 60d window (it adapts) and the
residual from a long window (250d / 1250d) where a persistent
cross-branch link is actually estimable. Result (559 rebalances,
baseline restarts at t=1250 so vols differ from the table above):

    L250-full +2.25% | L250-r1  +0.07%
    L1250-full +2.80% | L1250-r1 +0.20%

Verdict: no. The slow DENSE residual hurts even more than the fast one
(well-estimated but stale: a five-year average correlation regime is a
worse forecast of next month than the adaptive tree it disagrees with).
The slow rank-1 residual is exactly NEUTRAL (+0.07/+0.20%, within
noise): even the most persistent cross-branch pattern, measured over
five years, buys nothing at a monthly horizon.

## Bottom line (real data)

"The corr you left out" was tested in every form: raw fast residual,
denoised fast (r1/r2), slow dense, slow denoised. None beat gamma=0.
On real returns at monthly rebalance, the dendrogram IS the persistent,
exploitable part of the correlation structure; what the tree cannot
represent, next month does not honor. gamma* = 0: HRP is
self-sufficient, and its lossiness is not a bug but implicit shrinkage
-- the strongest possible endorsement of Peter's original framing,
just with the opposite practical conclusion to the lab's.

The transport machinery keeps its real job on real data: not a free
vol improver, but the repair/constraint engine (polish_race,
concentration constraints, belief surgery) -- moves you make because
you must, executed with minimal damage. And the side discovery stands:
rank-k+diag of the raw sample correlation is the best min-var input
tested here, on both synthetic and real (basket) data.

Data: 30_Industry_Portfolios_Daily.csv from Ken French's data library
(not committed; fetch the zip from mba.tuck.dartmouth.edu).
