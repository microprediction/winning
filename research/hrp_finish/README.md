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
