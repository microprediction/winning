# Track D: R&S under CRN + exact Thompson shares
(Adjudicated PURSUE narrow 2026-09-01; report in
../adjudications/rs_thompson.md. Adjacent: ../design/
rollout_control.md shares the KG/OCBA literature.)

## The claims under attack, in their words (to verify before quoting)
- VAPOR (arXiv:2311.13294): posterior probability of optimality
  requires "computing several complicated integrals with respect to
  the posterior" -- built variational surrogate instead. PDF saved
  in session tool-results; re-verify quote in our own copy.
- GSP (Ni et al., OR 2017, Sec. 2.5): "our procedure does not
  support the use of common random numbers."
- No procedure computes the exact joint PoM vector (adjudicated
  across KN, Rinott, GSP, KT/FBKT, OCBA, Chick-Inoue, correlated
  KG). The strongest such quote must be found and read in full.

## Decisive experiments
R&S: Negoescu-Frazier-Powell drug testbed (literally VV'+D, known
loadings, used by P3C arXiv:2402.02196) -- exact-PoM stopping and
allocation vs KN, P3C-KN, OCBA at matched PCS; report sample-size
ratio. Thompson: finite correlated Gaussian bandit -- exact-PoM
sampling vs VAPOR/VBOS vs TS on regret AND on VAPOR's own
approximation-gap metric. Synthetic data: no acquisition risk.

## Status
Not started. First action when opened: locate the Free-Wilson
drug-testbed specification (Negoescu-Frazier-Powell 2011) and
reproduce its factor structure.
