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
3. Robustness inversion: min-var's edge collapsed 5x under estimation noise
   (-4.07% -> -0.85%) while race-finishing held (-0.42% -> -0.32%). At T=45
   the free repair captures ~40% of what unstable optimization captures.

FINAL RECIPE: invert w_HRP under the cophenetic belief; re-price under the
rank-k+diag fit of the RAW sample covariance; gamma = 1; no other shrinkage.
