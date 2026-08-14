# Referee response: state as of 2026-08-14

## Done

Three substantive complaints, all acted on.

1. **Wrong null in the held-out test.** `luce_null.py` generates complete rankings
   from an exact Plackett-Luce process whose worths are the observed first-place
   shares, then reruns the identical five-fold pipeline. Eleven of twelve datasets
   exceed the null at p <= 0.005. Sports participation (n=130) does not: excess
   -0.0000, p = 0.51. Its gain is finite-sample shrinkage and it is now excluded
   from the paper's claims. Wherever n >~ 2000 the null is *negative*, so the old
   comparison with zero understated the effect. The properness sentence is deleted.

2. **Tail-thickness claim retracted.** Replaced by a proposition on the curvature of
   the reverse hazard r = f/F: log r concave => removal contracts the favourite's
   log odds; log r affine => equality, which is the Gumbel and recovers Yellott.
   Gaussian has (log r)'' = -Var(Z|Z<x) < 0. Verified in `hazard_contraction.py`.
   The referee's counterexample reproduces: on shares (.88,.07,.05) the Gaussian
   gives 0.0384 and t(3) gives 0.0555, so the ordering reverses with configuration.

3. **Saturated benchmark is not a ceiling.** E[L_sat] >= L_oracle, denominator too
   small, fraction biased upward. Retitled and restated. Median share corrected
   from 43% to 39% (the script took the upper of two middle values).

Also: loaders unified so all tables cover the same twelve collections (Netflix,
dots, puzzles recovered as separate rows); contraction table regenerated with
respondent-bootstrap intervals that recompute shares, calibration and both slopes;
occupational prestige restored under a shared add-alpha convention and it posts the
largest gain; all inputs committed under `research/human/data` at 1.9MB so nothing
reads from a temp directory; winner-orientation clash between Equation (2), the
pairwise formula, the figure and the lattice code resolved onto highest-wins.

**Ordered-outcome paragraph cut.** Pricing an exacta by removing the winner and
re-running is a sequential heuristic, not the race's ordering law. Measured against
the calibrated race by simulation it misprices cells by up to 3.4x, which exceeds
the effect it was meant to demonstrate.

## Running (launched 2026-08-14, output in this directory)

- `luce_null_200.txt` -- full 12 datasets x 2 alphas x 200 replicates. Hours. The
  only cell the paper currently takes from a shorter run is Sushi at alpha=0.5.
(`gain_by_size.txt` finished and is now Table 3 in the paper.)

## Landed after the status note was first written

Gain by menu size (`gain_by_size.txt`, now a table in the paper). The pairwise gain
is two to four times the all-subsets aggregate (Sushi +0.0412 against +0.0111; GSS
socialization +0.0130 against +0.0055), and the gain decays monotonically to exactly
zero at |T| = K, where nothing has been removed and the two accounts must coincide.
That last column is a pipeline check. So the aggregate is a conservative estimand,
and it is kept as the headline because it is fixed in advance rather than selected
after the fact.

## Not done

- Per-dataset flow table: source, K, raw and retained counts, tie handling, whether
  subsets are observed or ranking-induced, pooling, inclusion status, seed.
- The held-out bootstrap resamples respondent losses with the fitted training models
  held fixed, so it omits calibration uncertainty. The lambda-table bootstrap does
  recompute everything. This asymmetry should be stated or fixed.
- Consumer-products experiment is still not a primary result, though it is the only
  dataset observing real subset choice.
- Exacta and trifecta pricing redone against the exact ordering law.
- Nothing is pushed. The referee could not see `research/human` on main for that
  reason. A tagged commit or archival DOI is needed before circulation.
