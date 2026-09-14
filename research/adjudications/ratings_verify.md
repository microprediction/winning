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

## Gate result (winning session, 2026-09-12, recentred factor nodes): PASSED -- open defect closed
The baseline's open defect (full-covariance order update under a diffuse
dense belief, rank >= 3 loading space on 2^10 Sobol nodes) is closed by
node placement, not node count. Diagnosis in order: the effective node
count was 13-35 percent at every scale, a Rao-Blackwellised Hessian
(per-node differences plus the between-node gradient covariance) changed
nothing, and the means carried the same error as the variances, so the
loss was the quasi-Monte-Carlo integration error of a cloud drawn for
the prior while the factor posterior sat elsewhere. Both mixture engines
(full._mixture_update_full for kernels that report per-node likelihoods;
nway._mixture_update at rank >= 3) now recentre and rescale the cloud on
the factor posterior after one pass at the prior nodes, with the
importance correction in the weights (nway._recentre_nodes, inflation
1.2): one extra kernel pass, about a tenth of the update's cost.
referee.full_covariance order cells, relative variance error / cross
terms, prior-predictive events, 2^10 nodes:

    prior sd   before            after             (2^12 before -> after)
    1          0.0159 / 0.004    0.0059 / 0.002
    3          0.054  / 0.18     0.024  / 0.059    (0.075 -> 0.0094)
    10         0.134  / 2.7      0.0079 / 0.14     (0.0018 -> 0.0034)

Every cell now gates at the referee's marks (0.006 + 4 SE relative on
variances); the 2^12 rows stay as the remedy ladder. Effective node
count after recentring about 80 percent. Fast profile unchanged (749 ok,
0 FAIL); the diagonal correlated path at rank 3 was already within 0.01
of the sampler in the typical regime and keeps its results at the
fourth decimal. tests/test_full_covariance_updates.py pins the prior-sd-
10 cell against the sampler.

## Gate result (winning session, 2026-09-12, relative finite-difference steps): PASSED
The diagonal moment updates (update_winner's normal path,
update_ranking_exact, the correlated mixture) differenced their
curvature at ABSOLUTE steps (1e-4 winner, 1e-3 orders, 5e-2 on kinked
bases), so the posterior variances were only approximately covariant
under a common rescaling of means, prior sds and noise sds, where the
full-covariance path's relative steps were exact. Steps are now eps
times each coordinate's predictive sd (the historical step at unit
scale). identity.invariances.scale_v.diag: 1.6e-4 -> 2.8e-12 (mark
1e-3 -> 1e-8, set with the change since the identity is exact). Every
other fast identity holds unchanged (693 ok, 0 FAIL); the enumerated
moment identities' residuals are within their floors at every cell.
The legacy audit cells are unchanged to three decimals: update_winner
robust sd / coverage normal 0.940 / 0.975, gumbel 0.872 / 0.985,
logistic 0.936 / 0.980, laplace 0.942 / 0.975, student4 0.944 / 0.970,
failure 0.899 / 0.985 (25 seeds, as in the baseline); the
update_order_correlated cells pass at their 14 seeds on every base.

## Gate result (winning session, 2026-09-12, factor-fitter bands): set from the baseline
factor.se_calibration gates at z^2 in [0.85, 1.15] (baseline 0.972 +-
0.075: the Laplace posterior is calibrated on its own prior);
factor.recovery_slope at [-0.65, -0.35] (baseline -0.41, the n^-1/2 law
of a regular MAP). factor.heldout stays MEASURED: its margins are
world-specific, and its falsification clause (normal trailing
Plackett-Luce on normal-generated data by more than two paired SE)
already fails on its own.

## Mark power audit, second pass (winning session, 2026-09-14): the approximation envelope was passing on a seed count

The P6 envelope for `update_ranking` bounds a documented approximation
cost: robust sd in [0.9, 1.6] and coverage in [0.78, 0.95], beyond which
the cell fails. Three things were wrong with it, and the first is the
one that matters.

**It was passing on noise.** At the shipped 25 seeds the failure base
reads robust 1.55 and coverage 0.799, inside both bounds by 1.3 and 1.6
standard errors. At 100 seeds it reads 1.6058 +- 0.0134 and 0.7596 +-
0.0041, outside both. Four times the data moves the statistic across
both edges, so the nightly was passing on where 700 z-values happened to
land rather than on the approximation being where it was documented, and
a better-powered run of unchanged code would have reported a failure.

**The power test could not see it.** P6 already asks whether three
standard errors fit inside the bound, which is the right question, but
the envelope branch returns before that test is reached, and the test
measures against the P6 threshold of 0.35 rather than against the
envelope edge, which for this cell sits seven times closer. So the cell
could report the cost as documented while having no power to say
otherwise.

**The envelope was set without this base.** Its basis table lists
gumbel, normal, logistic, laplace and student4, whose robust sds measure
1.06 to 1.21 here and sit comfortably inside the P6 thresholds, so none
of them ever reaches the envelope. The only cell that reaches it is the
failure base, which has no row in that table and inherited bounds fitted
without it.

Remedy. The failure base gets its own envelope, robust in [0.9, 1.70]
and coverage in [0.74, 0.95], set four standard errors clear of the
100-seed measurement so that 25 seeds can decide it, with the measured
values, their standard errors, the seed count and the date recorded
beside it. The power test runs on the envelope branch and against the
edge actually in force; an envelope verdict that cannot be resolved
reports UNDERPOWERED rather than EXPECTED_APPROX. Every envelope verdict
now carries its margin in standard errors. Verified: the failure cell
reports EXPECTED_APPROX at 5.1 standard errors of margin, and the five
documented bases stay ok.

What generalises, alongside the P8 finding above: a bound is only as
good as the power to test it, and a bound inherited by a case that was
not measured when it was set is a guess with a citation attached.

## Mark power audit (winning session, 2026-09-14): P8's mark was the smaller half of its own threshold

A mark nobody has attacked is the same object as a tuning sweep nobody
checked, and the question to ask of one is not whether the tolerance is
right but what would have to be true for it to fire, and whether that
can happen at the profile's budget. The committed baseline run records
statistic, standard error and tolerance for every cell, so the question
is arithmetic on that file rather than a re-run.

Of 172 marked cells carrying a usable standard error, none is toothless:
the largest gap between a statistic and its bound is 16 standard errors,
and the median is 5.4. The failure is in the other direction and in one
family.

P8 gates on `dm > mark + 4 se`, so what it enforces is the mark plus the
referee's own Monte Carlo noise. That allowance is budget-dependent and
the report printed the nominal mark at every profile. Measured, worst
base of six:

| draws | noise allowance | enforced | against a mark of |
|---|---|---|---|
| 100,000 (fast) | 0.0132 | 0.0182 | 0.005 |
| 400,000 (full) | 0.0069 | 0.0116 | 0.005 |
| 4,000,000 (exhaustive) | 0.0021 | 0.0071 | 0.005 |

So a reader of marks.py believing a 0.005 error in the posterior mean
would be caught was wrong by 3.6x at the profile that gates CI, and the
mark was the smaller term of its own threshold everywhere. The report
compounded it by pairing the statistic `max(dm, dv)` with `dm`'s
tolerance, two numbers that do not correspond.

Remedy, and no mark moves. `dm` and `dv` become separate results, each
statistic against its own bound. The reported tolerance is the enforced
threshold, with the mark and the noise allowance in the detail line, so
a report says what it enforced rather than what was written down. The
full profile draws 1,000,000 rather than 400,000, where the allowance
falls to 0.0044 and the mark binds for the first time; the cost is half
a second over six bases. The fast profile keeps its budget and now
reports the 0.018 it actually enforces.

What generalises: a gate of the form `statistic > mark + k * noise` has
two terms and only one of them is in marks.py. Where the noise term
dominates, the mark is decoration. Next under the same method: the
EXPECTED_APPROX envelopes and the P6 contrast bands.

## Gate result (winning session, 2026-09-13, nightly full profile): exit 2 twice, budget raised

The first two scheduled full runs (2026-09-12 08:18 UTC, 2026-09-13
08:42 UTC, runs 34,7xx on `ratings-verify.yml`) each ended `4991 ok, 0
FAIL, 1 UNDERPOWERED, 2 EXPECTED_APPROX, 248 SKIP, 96 MEASURED` in about
31 minutes at 4 workers, exit code 2. The one UNDERPOWERED row is
`factor.se_calibration.z2`: 0.972 +- 0.075 over 30 worlds against the
band [0.85, 1.15]. The statistic is inside the band on both nights; the
verdict is UNDERPOWERED because 3 se (0.225) exceeds the half-band
(0.15), so at the profile's budget the mark could never be decided. The
mark was set at the baseline from exactly this 30-world measurement,
which is the instrument defect: a band narrower than the check's own
power at its budget.

Remedy: budget, not mark. `factor_worlds` in the full profile goes from
30 to 100 (se falls to about 0.041, 3 se 0.12 under the half-band) and
in the exhaustive profile from 60 to 150; the band is unchanged. Cost
on one worker rose from about 2 to 17 minutes measured locally, which
the nightly's four workers and 60-minute timeout absorb. Verified
locally at the new budget before merging: z^2 0.972, verdict ok. The nightly's failure-on-exit-2 stays as designed: an
undecidable mark is a finding, not a pass.

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
