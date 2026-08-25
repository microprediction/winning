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
