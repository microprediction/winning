# Shifted control variates for general-covariance Gaussian share inversion

The experiment asks whether winner shares of a hard race `U = mu + eps`,
`eps ~ N(0, Sigma)` with arbitrary `Sigma`, can be estimated and inverted
faster by coupling every Monte Carlo draw to an **easy surrogate race whose
location parameters are shifted so its winner shares equal the target
shares** `p*`, so that only the discrepancy between the races is left to
Monte Carlo:

    p_Sigma(mu) - p* = E[ e_W(mu) - e_V ],        E[e_V] = p*  by construction.

Verdict up front: **the idea works exactly to the extent that the surrogate
covariance reproduces the real one.** When `Sigma` is (approximately) low
rank plus diagonal — factor structure, few clusters, strong spectral decay,
one dominant correlated block — a rank-4/8 surrogate calibrated to `p*`
gives winner agreement `P(W=V)` of 0.9–0.99 and, combined with one-factor
Rao–Blackwellisation, residual-variance reductions of 10^2–10^4 (median ~40
at n=50, best ~10^4), holding up at n=1000 (factor family: VRF ~900–1400 at
`P(W=V) = 0.99`; that is §24's "strongest possible result" scenario, and it
occurs). When `Sigma` is genuinely high-dimensional — dense Wishart,
near-singular half-rank, clusters whose count grows with n — no cheap
surrogate agrees with the race often enough and the method buys ~1–2x at
2x cost, i.e. nothing. The single number `P(W=V)` predicts everything else
(Plot 1), so it is the right cheap pilot diagnostic to decide whether to use
the machinery on a given `Sigma`.

## Where things live

    envelope_fast.py     numba one-factor envelope kernel: exact conditional shares
                         q(eta), conditional Laplacian Jacobian, RB shares
                         (algorithm of research/qpo/envelope.py, re-validated here
                         against brute force + finite differences)
    problems.py          covariance families x ability regimes; reference shares p*
                         by RB envelope with 100k-400k draws, cached in cache/
    references.py        surrogate races (logit; iid/diag/low-rank probit via
                         winning.factor.core forward+inverse), couplings
                         (common-z, sym/chol/eig square roots, Procrustes, Cayley
                         hill-climb over orthogonal Q)
    estimators.py        the estimators: one-hot CV, RB, RB+CV, multi-control
                         regression; per-draw variance/agreement diagnostics
    invert.py            sample-average Newton with fixed coupled draws,
                         soft-thresholded residuals (2 se), exact reference Jacobian
    run_agreement.py     Section 21 critical test -> results/agreement.csv
    run_inversion.py     Sections 14/17/19 -> results/inversion.csv
    run_distance.py      Sections 8/9, Plot 7 -> results/distance.csv
    run_jacobian.py      Section 13 -> results/jacobian.csv
    run_variance_vs_M.py Plot 4 -> results/variance_vs_M.csv
    make_figures.py      Plots 1-8 -> figures/
    results/summary.txt  the tables cited below (regenerate: see git log of this file)

Conventions: max-wins throughout (`winning.factor.core` is min-wins; sign
flip at the boundary). All samplers use the projected covariance
`Sigma_c = P Sigma P` — common shocks cannot change the winner. Everything
ran with 92 problems: 10 covariance families x 4 share regimes x
n in {50, 250} plus an n=1000 subset, seeds fixed, `OMP_NUM_THREADS=3`.

## The identity, verified

`tr Cov(r_hat) * M` is flat in M for every estimator (variance_vs_M.csv) and
the mean control residual at `mu*` is statistically zero (`|bias|_1 / se`
median 1.2) **except** where VRF is so large that two floors show through:
the p* reference-share accuracy (median `|se(p*)|_1 ~ 0.01` at 100k-400k RB
draws) and the surrogate's lattice-quadrature calibration (~1e-3 relative,
2^9 Sobol nodes x 501 lattice points). On the factor family the coupled
estimator hits those floors at M ~ 100 draws — the estimator is then better
than the ground truth it is being scored against.

## Findings by hypothesis

**H4 (agreement predicts performance): confirmed, and it is the headline.**
Plot 1: across ~1000 (problem, estimator) pairs the one-hot VRF sits on
`(1 - sum p*^2) / (2 (1 - P(W=V)))`, the exact relation implied by
`sum_i Var(e_W,i - e_V,i) = 2 P(W != V)` at the solution.

**H2 (covariance proximity matters): confirmed; it is nearly the whole story.**
Median VRF of the target-share-matched control by family (best coupling,
rank 8 + RB): factor ~1200, clustered5 ~160, asymmetric ~72, spectral0.25 ~28,
spectral2 ~19, dense ~1.6, nearsingular ~1.5. Same-share logit and iid
references — identical marginal shares, wrong covariance — deliver VRF ~1.
Rank ladder (Plot 6): median one-hot VRF 1.4 / 1.9 / 3.7 / 5.1 / 10.5 at
rank 1/2/4/8/16 (n=50). But rank must track the covariance's effective rank,
not n: 'clustered' with n/10 clusters defeats a rank-8 surrogate at n=1000
(VRF 1.2) while 'clustered5' with 5 clusters gives VRF 90–190 there.

**H3 (coupling matters): confirmed.** Independent coupling is the negative
control at VRF 0.50 (exactly the 2x-variance prediction). Median over
problems (rank-4 reference): common sym-sqrt 1.7, Procrustes-rotated
sym-sqrt 3.0-4.0. The symmetric square root beats Cholesky and eigen
factors even after Procrustes (3.95 vs 2.3). The direct winner-agreement
hill-climb over orthogonal Q (80 Cayley steps from the Procrustes start)
does NOT improve out-of-sample (median 0.985x) — it overfits its pilot;
Procrustes is the practical optimum here.

**H1 (share matching matters): weakly supported, with a caveat the spec did
not anticipate.** For the logit reference the temperature cancels exactly in
the share-matched control — `argmax(tau log p* + tau g)` is tau-invariant —
so Plot 3's tau sweep only exists for the same-mu logit (where higher tau is
strictly worse). Share matching is what makes `E[e_V]` known exactly and
free, and against a *badly located* reference it matters enormously
(same-mu logit degrades inversions 6x at M=64). But once the reference is
covariance-matched and the shift is carried with mu (`nu = mu + a`), a
share-matched location adds little beyond the location match itself: near
the solution `lowrank_shift` and `lowrank_samemu` invert identically.

**H5 (RB and shifted controls compound): confirmed.** Median VRF at n=50:
one-hot lowrank8 5.1, RB alone 1.6, RB+lowrank8 43.9. The compounding is
multiplicative-ish because the RB estimator replaces the 0/2 disagreement
indicator with the difference of two smooth conditional-share vectors
(Plot 8). Falsification criterion 6 (RB alone dominates) is rejected: RB
alone never exceeds ~4.

## The moving control (Sections 8-9, Plot 7)

The fixed target-share control dies away from the solution: VRF crosses 1 at
RMS distance ~0.4-0.8 ability scales and is 0.34 at 1.6. The repair that
works is not re-anchoring from a Monte Carlo pilot — a 512-draw pilot's
location noise caps the local control at VRF ~70 where the ideal shift gives
~2700 (factor, n=50) — but carrying the *shift* `a = nu-tilde - mu-bar`
computed once from a good anchor: that "stale" control loses almost nothing
out to 1.6 scales (factor: >500 at t=0.8). The two-control regression
(fixed + local, betas fitted on a pilot) recovers whichever is better
automatically and is the recommended construction; the fitted beta also
roughly doubles the RB control's VRF near the solution (median 1.9x) and is
free.

## Jacobian (Section 13)

The coupled Jacobian correction fails: envelope-Laplacian draws are heavy
tailed (edge weights `phi(tau)/|b_j - b_i|`), the correction's variance
exceeds its covariance with the target, and `J_0 + mean(J^m - J_0^m)` is
*worse* than the target-only average at every M tried. What wins is simpler:
the **exact Jacobian of the calibrated surrogate**, `J_0(nu*)`, is 2-10%
from `J_Sigma(mu*)` in Frobenius norm — closer than a 256-draw envelope
estimate — and costs one reference computation. All inversions here use it.

## Inversion at matched budgets (Sections 14/19, Plots 5, 8)

All methods share: start `mu0 = nu*` (the surrogate answer), fixed coupled
draws (CRN across iterations), `J_0` Jacobian, soft-thresholded residual
(components within 2 se of zero are treated as resolved — without this,
one spurious count on a rare entrant produces a maximal Newton step and raw
MC *destroys* the surrogate answer it started from). 0 failures in 1164
inversions.

The structure of the results: inversion error in share space equals the
estimator's noise level `~ sum_i sqrt(Var_i / M)`, so VRF converts directly
into how small M can be before Monte Carlo refinement beats the surrogate.
On dense Sigma nothing beats the surrogate until M ~ 1000 and then only the
RB+CV estimator (median share-L1 ratio 0.91 at M=4096); on the clustered
families with the rank-8 surrogate the starting point is already at the p*
accuracy floor and refinement is a wash (ratios 0.98-1.0); raw and
badly-located estimators are the only ones that make things *worse* at small
M. The surrogate inversion itself — a deterministic lattice computation —
does most of the work everywhere: median share-L1 0.033 (rank 4) / 0.023
(rank 8) with zero Gaussian draws.

## Normalized efficacy (measured, not flop-counted)

Per-draw wall clock against raw winner counting (bench_cost, factor family):
at n=250 a coupled one-hot draw is 6.5x and the RB+control draw 11.4x — the
diagnostic bookkeeping and the envelope loop dominate the tiny 2us BLAS
draw. At n=1000 the O(n^2) matvecs take over and the ratios settle to ~2.1x
(one-hot) and 5-7x (RB+control; the envelope loop is pricier when many lines
reach the envelope, e.g. 7.4x on clustered5). Net at n=1000: factor VRF 890
/ 5.1 ~ 175x, clustered5 VRF 193 / 7.4 ~ 26x genuine wall-clock speedup, and the one-time surrogate
calibration (~7 min at n=1000, rank 4-8) amortizes as soon as the raw
budget it replaces exceeds a few seconds. Across all 92 problems the
cost-adjusted (VRF/2) view: RB+rank8 is worth using on 80% of problems,
>3x on 65%, >10x on 47%; one-hot rank8 on 64% / 38% / 26%; plain RB is
never harmful but never exceeds ~10x.

**Rule of thumb for applicability** (measured on the test covariances):
fit BB' + D to P Sigma P at rank 8 and look at the OFF-DIAGONAL energy the
fit fails to explain — the common mode is projected out and the diagonal is
free, so neither counts against the rank. Uncopied share vs outcome:
factor 0% -> VRF ~1000; clustered5 0% -> ~100-200; spectral2 2% -> ~20-50;
spectral0.25 68% but off-diagonals tiny in absolute terms -> ~5-30;
clustered(n/10 groups) 81% -> ~1; dense 97% -> ~1. Under ~10-20% leftover
(or near-independence) the machinery pays; above ~50% skip it. The
4096-draw coupled pilot measuring P(W=V) is the direct referee.

## Costs, honestly (falsification checklist, Section 22)

1. Target-share calibration does raise agreement vs an uncalibrated
   reference of the same family — but location match via a carried shift is
   what matters; calibration is the cheap way to get it. Partially upheld.
2. Agreement stays near chance for genuinely high-rank Sigma (dense,
   nearsingular, growing clusters) at every rank tried. **Method falsified
   for those families** — and `P(W=V)` on a 4096-draw pilot tells you in
   advance.
3. Reference cost: the rank-r inversion `W_Sigma0^{-1}(p*)` took (median)
   13s/50s/190s at n=50/250/1000 for rank 4-8 — amortised once per problem,
   negligible against any serious Monte Carlo budget, but the *coupled
   draw* costs 2x a raw draw (two dense n x n matvecs), so quote
   cost-adjusted VRF = VRF/2. Cost-adjusted medians for RB+rank8:
   22 / 3.8 / 1.75 at n=50/250/1000 over ALL families — the median n=1000
   problem is not worth it; the right families very much are.
4. Variance reduction away from the solution: fixed control dies, carried
   shift + regression combination repairs it. Upheld-then-repaired.
5. Coupling estimation cost: Procrustes is one SVD; the expensive
   winner-agreement optimisation adds nothing. No issue.
6. RB alone does not dominate the combination. Rejected.
7. Inversion accuracy at fixed cost improves only where VRF is large AND
   M is large enough to resolve the surrogate's residual; elsewhere the
   surrogate answer alone is the best use of the budget.

## Reproduce

    OMP_NUM_THREADS=3 NUMBA_NUM_THREADS=2 python -u run_agreement.py --n 50 --ranks 1 2 4 8 --qopt
    ... run_agreement.py --n 250 --ranks 1 2 4 8
    ... run_agreement.py --n 1000 --families clustered5 factor spectral1 spectral2 dense --regimes diffuse moderate --ranks 1 2 4 8
    ... run_inversion.py --n 50 --families dense factor clustered spectral1 --regimes diffuse moderate --M 64 256 1024 4096
    ... run_inversion.py --n 50 --families clustered clustered5 --rank 8 --tag _r8 --methods raw lowrank_shift lowrank_samemu rb rb_lowrank_shift rb_lowrank_samemu multi rb_multi --M 64 256 1024 4096
    ... run_variance_vs_M.py --n 50 && python -u run_jacobian.py --n 50 --families dense factor clustered
    ... run_distance.py --n 50 --families dense factor clustered --regimes diffuse moderate
    python make_figures.py

Everything is resume-safe (rows keyed in the CSVs are skipped on rerun).
Total compute for the committed results: ~2.5 hours on 3 cores.

## What I would build from this

For a production general-covariance inverter: fit `Sigma0 = BB' + D` at the
effective rank (scree of `P Sigma P`), invert the surrogate exactly (that
answer is already share-L1 ~0.02-0.03), measure `P(W=V)` on a 4096-draw
Procrustes-coupled pilot, and only if it clears ~0.9 spend Monte Carlo on
the RB + carried-shift control with regression betas, stopping at the p*
noise floor. If the pilot agreement is poor, the honest options are raw
RB Monte Carlo or a better surrogate family — the shift cannot rescue a
covariance mismatch.

---

## The exotics extension moved out

The ordered-outcome work that grew out of this experiment -- exacta/trifecta
pricing, the cavity closure, the identification hierarchy (win -> mu,
place -> sigma, exacta -> correlation) and the cavity reading of the ability
transform -- now lives in its own repository:

    https://github.com/microprediction/exotics   (private)

It vendors `envelope_fast.py` and `references.py` from here as dependencies, so
those two files are shared; everything else in this directory is the shifted
control-variate experiment and stands alone.


## Negative result (2026-08-26): Gumbel surrogates as control variates

The tempting idea: Gumbel races have closed-form shares (softmax; mixed
logit and nested logit for the factor and tree twins), so couple each
Gaussian draw to a Gumbel draw through common uniforms and Monte Carlo only
the discrepancy. Measured at N = 100 via the VRF identity
(1 - sum p^2) / (2 (1 - P(same winner))):

    factor Sigma       P(same) 0.386   VRF 0.78   (hurts)
    dense Sigma        P(same) 0.457   VRF 0.88   (hurts)
    independent Sigma  P(same) 0.555   VRF 1.06   (nothing)

Even under independence -- where the Gumbel twin is exact in distribution --
quantile coupling agrees on the winner barely half the time, because the
argmax is decided in the right tail, exactly where the Gaussian and Gumbel
quantile maps disagree most. Same lesson as the main study from a new angle:
coupling quality is everything, and SHAPE mismatch destroys it where winners
are decided. The surrogate that worked here was same-shape with calibrated
location; closed-form shares are worthless as a CV if the shape differs.
