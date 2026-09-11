# Ledger: winning.ratings verification runs

The ratings layer's verification suite lives in `winning/ratings/verify`
(run it with `python -m winning.ratings.verify --profile full --workers 4
--out reports/`). This file records the ADJUDICATED runs: the baseline,
and the before/after pair around every fix or documented approximation,
each citing its committed report under `ratings_verify/runs/`. Nightly
runs live as CI artifacts and are not recorded here.

Rules, in force from the first entry:

- A mark (a tolerance, seed count or envelope) lives in
  `winning/ratings/verify/marks.py` and nowhere else, with who set it,
  when, and on what basis. The report records that file's hash.
- A mark is set before the fix it will judge exists; a fix commit does
  not touch marks.py. A check with no mark reports MEASURED and never
  gates; the adjudicator turns a measurement into a mark or into an
  EXPECTED_APPROX envelope, with the number and a citation, never into
  silence.
- Referee configurations beyond the fixed rows are drawn from
  `marks.CONFIG_SEED`, bumped by the adjudicator when a "before" run is
  recorded, so the implementer neither picks nor sees them first.
- UNDERPOWERED is a verdict distinct from ok and fails the run.

Entry format (after research/adjudications/laplace_convolution_shortcut.md):

    ## Gate result (<who>, <date>, <profile>, <commit>): PASSED | FAILED | UNDERPOWERED
    run: ratings_verify/runs/<date>_<sha7>_<profile>.json
    <rows that changed, before -> after, with the check names>
    <what the result rules out; residuals adjudicated NOT a defect>

## Gate result (winning session, 2026-09-11, F2 predict-versus-evidence): PASSED
before: baseline run at 25121e0 (identity.predict_evidence.* MEASURED);
after: commit b018af1.
`AbilityTracker.predict` and `predict_race` priced non-normal bases at
D = beta2 + v (the convolution shortcut the updates dropped in 4cda4f1)
while `observe` priced the belief convolved with the base noise. The
mark was set from theory before the fix (lattice-mass normalisation,
1e-4). identity.predict_evidence, |log p(winner) - evidence increment|,
belief variance 0.5-1.5 against unit noise:

    base      rho   before    after
    gumbel    0.0   0.0568    3.2e-6
    gumbel    0.4   0.1163    2.7e-6
    laplace   0.0   0.0963    4.3e-6
    laplace   0.4   0.0747    3.6e-6
    logistic  0.0   0.0156    3.6e-6
    logistic  0.4   0.0165    3.1e-6
    normal    0.0   2e-16     2e-16   (regression row, unchanged)
    normal    0.4   3.4e-6    1.3e-6

The new `nway.predictive_win_probabilities` matches brute-force
simulation of the convolution model within Monte Carlo error on a
diagonal Laplace belief (max 2e-4, MC se 4e-4) and on a full-covariance
Student-t belief with loadings (5e-4). Rules out: any remaining
model mismatch between prediction and update on the tested bases; what
is left is lattice mass.

## Gate result (winning session, 2026-09-11, F1 full-state rate_history): PASSED
before: the pre-fix module (git HEAD~1 history.py, loaded beside the
fixed one, same worlds); after: this commit.
`rate_history` gathered the entrants' block of S, updated it and wrote
it back, so cross-covariances between a race's entrants and everyone
else never moved. Every observation now lifts through the full state
via a selection matrix (the teams kernels; `update_team_market_full`
added). The mark (identity.history_conjugate.exact, 1e-8) was set from
theory: a conjugate full-state filter equals the closed-form joint
posterior. Static conjugate world, 10 entities, 24 races of 4,
lengths_scale 1:

    statistic                              before     after
    max |m - m_joint|                      0.165      2e-15
    max |S - S_joint|                      0.065      1.5e-15
    order dependence of m (permutation)    0.244      3e-15
    order dependence of logZ (nats)        0.94       0.0
    contrast z^2, 40 worlds (12 x 60 x 5)  1.32+-0.09 0.963+-0.061
    contrast coverage                      0.913      0.956

CORRECTION to earlier session notes: numbers quoted before this entry
(contrast z^2 18, coverage 0.37, means off 1.2) came from a harness
that omitted lengths_scale=1.0 (the default 0.2 scales score
observations by a fifth); the defect was moderate, as tabled here, not
catastrophic. calibration.history.contrast now gates at the powered
band [0.85, 1.15] (R = 80); the exact identity is the sharp test.
Residual, MEASURED: rate_history's order-observation path is an ADF
projection and stays order-dependent (mean spread 0.022 under a
permutation of 12 races).

## Open items entering the ledger as MEASURED (2026-09-11)

- identity.invariances.scale_v on the diagonal paths: absolute
  finite-difference steps (1e-4 winner, 1e-3 orders) make the
  variances only approximately scale-covariant (1.6e-4 at c = 0.1 on
  logistic orders); the full-covariance path's relative steps are
  covariant to 1e-8. Candidate: relative steps in nway.
- audit.P6.*.marginal: the field-centred marginal statistic of every
  driver, reported next to the pairwise-contrast statistic the port
  gates on.
- audit.P7.*.variances: variance spread across paths on identical
  evidence (curvature methods differ).
- identity.reductions.*_full_vs_correlated_V: full-covariance vs
  correlated with loadings (Gauss-Hermite 9 vs 7 nodes), 5.5e-5 and
  1.3e-4, marked at 1e-3.
- identity.history_conjugate.orders_order_dependence: ADF order
  dependence of rate_history on order observations (0.022).
- factor.se_calibration, factor.recovery_slope, factor.heldout: the
  Laplace posterior calibrated on its prior (z^2 0.97 +- 0.07), recovery
  slope -0.41, held-out margins; bands to be set from the baseline.
- calibration.tracker.*.marginal: the diagonal tracker's marginals are
  overconfident by the unidentified common level (z^2 2.2-2.5), by
  construction; contrasts gate.
