# Thurstonian recalibration: (V, D) from win-only classifier data

Gatekeeper experiment for the proposed layer: freeze any classifier,
read its logits as race abilities mu_t(x), fit one shared factor
covariance (V, D) post hoc by exact maximum likelihood on held-out
winners, and replace the softmax with the exact correlated race.

`run_win_only_identification.py`: K = 12 classes, rank-1 truth with
centered loadings and heteroskedastic D, menus mu_t ~ N(0, 4) iid per
example, winners sampled from the true race, exact-gradient ML via
torch autograd on the (k+1)-dim conditioning formula (anchored against
winning.factor to 9.3e-5 on classes above 1e-4; GH tails lose relative
precision below that, irrelevant to the likelihood since winners arrive
at rate p).

Result (results_identification.csv):

    T       |cos(v)|  med|dlogD|  contrast err  gap vs truth  gain vs diagonal
    1,000   0.846     0.259       0.433         0.0430/race   -0.0323/race
    5,000   0.941     0.130       0.286         0.0041/race   +0.0018/race
    20,000  0.990     0.042       0.114         0.0005/race   +0.0043/race

VERDICT: identified. Win-only data with per-example menu variation
recovers the covariance, monotonically in T at roughly root-T rates,
and the fitted correlated model beats the diagonal-only (independent
heteroskedastic) fit out of sample from T ~ 5,000. Below that the
correlated fit overfits (negative gain at T = 1,000): the layer needs
a few hundred winners per class at free per-class loadings.

Consequences:
- The layer is viable as a paper: "any classifier, correlated
  uncertainty, post hoc, exact."
- At K = 1000 with a 50k validation set (50 winners/class), free
  per-class (v_i, D_i) will be data-starved; the fix is the BLP trick
  in reverse -- tie V to fixed class embeddings (V = E A with E from
  the classifier's own head, A small) so the parameter count drops
  from O(K) to O(rank x embed-dim).
- Next: scale run at K = 100-1000 with embedding-tied loadings, then a
  real public checkpoint, then set-efficiency against conformal
  baselines (conformalize the race score: guarantee kept, sets smaller
  if the ordering is better).

## Application scouting digest (2026-08-25, three fronts)

Details with verbatim quotes: `apps-nlp-bio.md`,
`apps-forecasting-medicine.md`, `apps-industrial.md`.

**The unifying pitch, in the field's own words.** Between temperature
scaling (one parameter, cannot express between-class structure) and
matrix scaling (Guo et al.: any calibration model "with tens of
thousands (or more) parameters will overfit") there is a missing rung.
Low-rank-plus-diagonal is that rung, and unlike every incumbent it
produces a JOINT distribution -- restricted menus, top-k sets,
co-confusability -- not a map on marginals.

**The selection criterion that explains every ranking: does the menu
repeat?** Pooled ML for (V, D) needs many races against the same N
outcomes (or embeddings to tie across menus). Classifiers are the
ideal case (fixed menu, varying mu). Metaculus-style question sets
fail it (vivid IIA quotes, but menus never repeat).

Cross-front shortlist:
1. LLM multiple-choice calibration (semantic entropy is a hard-clustered
   special case of a fitted V; "temperature scaling significantly
   deteriorates calibration"; free data via lm-eval-harness).
2. Verbal autopsy cause-of-death (34 fixed causes, frozen scorers,
   PHMRC gold standard in one R call, restricted cause-lists native to
   practice, currently renormalized).
3. Winter precipitation type (rank-2 truth is KNOWN PHYSICS -- melting
   and refreezing layers; a fitted V that reproduces thermodynamics
   from labels alone is a spectacular figure; mPING + RAP open).
4. Cell-type annotation (organ-restricted menus; field admits bad
   calibration and broken ontology hierarchies).
5. Crop-type / remote sensing (phenology factors, region menus,
   public benchmarks).
6. Fault diagnosis and malware families (drift refits, top-k
   troubleshooting sets, shared-codebase factors).
Demoted with reasons in the files: variant nowcasting (correlated
modelling already published), forecast aggregation (menus never
repeat), ENSO/terciles (N = 3, ordered).
