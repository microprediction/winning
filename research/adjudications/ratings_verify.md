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

## Open items entering the ledger as MEASURED (2026-09-11)

- identity.predict_evidence on non-normal bases: AbilityTracker.predict
  and predict_race price non-normal bases at D = beta2 + v (the
  convolution shortcut the updates no longer use); measured 0.016
  (logistic) to 0.116 (gumbel, blocked) nats on the observed winner
  with belief variance comparable to the noise. Plan step 10 (F2).
- identity.invariances.scale_v on the diagonal paths: absolute
  finite-difference steps (1e-4 winner, 1e-3 orders) make the
  variances only approximately scale-covariant (1.6e-4 at c = 0.1 on
  logistic orders); the full-covariance path's relative steps are
  covariant to 1e-8. Candidate: relative steps in nway.
- audit.P6.*.contrast: pairwise-contrast calibration of every driver
  (F3), reported next to the field-centred marginal statistic the
  bandits audit gated on.
- audit.P7.*.variances: variance spread across paths on identical
  evidence (curvature methods differ).
- identity.reductions.*_full_vs_correlated_V: full-covariance vs
  correlated with loadings (Gauss-Hermite 9 vs 7 nodes), 5.5e-5 and
  1.3e-4.
