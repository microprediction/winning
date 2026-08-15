# Ranked-season results: exact ordered-statistics updates vs TrueSkill

Exact update: `winning.ratings.update_ranking_exact` (backward lattice
recursion over the joint ordered-statistics likelihood, analytic adjoint
gradient, O(nL) per race). Field 20 from a pool of 200, full finish order
observed, 1500 races, seed 7.

The homogeneous season is TrueSkill's home model and the exact update
reproduces it to about 1e-3 per rating: TrueSkill's EP is essentially
exact on its own generative model, so nothing can beat it there. The
heteroskedastic season gives even players noise sd 0.5 and odd players
sd 1.5; the exact update takes per-player scales natively while the
trueskill package is restricted to one global beta, and the gap widens
with data.

```
Homogeneous noise (TrueSkill's home model; expect a tie)
 races  exact rho  TS rho  exact rmse  TS rmse
   100      0.925   0.925       0.327    0.327
   300      0.969   0.969       0.203    0.203
   700      0.987   0.987       0.124    0.124
  1500      0.993   0.993       0.085    0.085

Heteroskedastic noise (per-player scales; TrueSkill restricted to one beta)
 races  exact rho  TS rho  exact rmse  TS rmse
   100      0.911   0.911       0.349    0.366
   300      0.966   0.964       0.220    0.241
   700      0.985   0.983       0.140    0.175
  1500      0.994   0.992       0.094    0.145
```
