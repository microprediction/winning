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

## Gate result (winning session, 2026-09-11, full profile, 25121e0): BASELINE
run: ratings_verify/runs/2026-09-11_25121e0_full.json (28.9 min, 3 workers,
1 thread each). Counts: 4952 ok, 28 FAIL, 0 UNDERPOWERED, 2
EXPECTED_APPROX (update_ranking, documented), 248 SKIP (Gaussian-only
update_winner_full under non-normal bases; the student-t correlated
cells identical to the diagonal path at V = 0; diagonal drivers at
V != 0), 102 MEASURED. The legacy audit group reproduces the bandits
digits. The 28 failures adjudicate into three items:

1. identity.moments, 21 rows, curve bases at the extreme regimes
   (noise variance 4 against prior variance 0.25; prior variance 9;
   Student-t tails): statistics 5.1e-5 to 6.7e-4 against floors of 5e-5
   (winners) and 5e-4 (orders) piloted at unit noise. The residuals are
   lattice mass -- I3 (normalisation) moves with I2 and the raw and
   clamped I2 coincide -- not curvature. Adjudicated: curve floors
   doubled (winners 1e-4, orders 1e-3), 1.5x the full-grid maximum;
   marks.py records the basis.
2. audit.regime.P1_P5, 2 rows: P4 (correlation with the truth after 40
   winner-only updates of 8 arms) at 0.14 (normal, noise variance 4)
   and 0.32 (gumbel, prior variance 0.25). With 8 arms the statistic's
   SE is ~0.35 and a low-information regime legitimately sits below
   0.5; the identical means through the independent path did not fail
   only because their world differs. Adjudicated: P4 gates at the
   baseline regime (the bandits statistic); off-baseline it is reported.
3. referee.full_covariance, 5 rows: the ORDER update with a dense
   rank-2 belief plus one-factor loadings (belief split rank 3, hence
   2^10 Sobol nodes) at prior sd 3 and 10 against the rejection-sampled
   posterior. Part referee, part engine. Referee: the cell conditioned
   on a fixed order that is near-impossible at a spread of ten standard
   deviations (732 accepted draws); it now conditions on a prior-
   predictive event and declares fewer than 2000 accepted draws
   UNDERPOWERED. Engine, on the corrected referee (63k-150k accepted):
   relative variance error 0.063 at sd 3 and 0.139 at sd 10, cross
   terms 15 and 8 percent of their scale, means within 1.6 percent of
   the prior sd; the winner update is fine at every sd. The Hessian
   step is not the cause (eps_rel 0.05..0.3 changes nothing); the node
   count is, erratically: 2^12 nodes give dv/v 0.0018 at sd 10 but
   0.075 at sd 3 (cross terms 0.19 of 1.2); 2^14 give 0.004 and 0.010.
   OPEN DEFECT: the full-covariance order update's node budget
   (full._mixture_nodes, 2^10 scrambled Sobol at rank >= 3) is too
   small once the belief dwarfs the noise in a rank >= 3 loading
   space, and simply raising it converges unevenly. Candidate remedy:
   a recentred or adaptive rule keyed to the belief/noise ratio (as
   races.py escalates sharp factors) rather than a fixed budget. Until
   then the sd 3 and sd 10 order cells report MEASURED at both 2^10 and
   2^12 nodes; the sd 1 cells and every winner cell gate.

Other MEASURED rows in the baseline, for the record: the diagonal
tracker's marginals (z^2 2.2-2.5, by construction), audit P6 marginal
and P7 variance statistics, ADF order dependence of rate_history on
order observations (0.022), factor.se_calibration 0.97 +- 0.07 (band to
set: [0.85, 1.15] proposed), factor.recovery_slope -0.41 (band [-0.65,
-0.35] proposed), factor.heldout margins (normal beats PL by 0.036 +-
0.016 on normal-generated data; beats uniform by 2.25).

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
