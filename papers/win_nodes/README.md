# Win Nodes for Bayes Nets

**Draft** (2026-09-06, first version written in-session). Thesis: the
n-ary win/argmax node as an exact, linear-work, differentiable query
on Gaussian graphical models -- factor structure via the shared
survival field (with the graph-Laplacian Jacobian), chains/bands via
restricted transfer-operator passes -- plus the exact observed-win
likelihood message of which TrueSkill's greater-than factor is the
n=2 case. Numbers trace to committed seeded scripts: research/tridiagonal
exp1-exp4 (numerical checks) and exp6_career_peak (the worked
example); software claims trace to the parity harness and package
tests. The factor case is CEDED to papers/exact_pom, which developed
it first and further; this paper keeps the chain and band layer and
the observed win node that feeds it.

Before any submission:
- the house quote-verification pass (papers/quote-verification.md
  culture) on Bolin-Lindgren, Ridgway, Muller-Nesterov-Shikhman and
  the TrueSkill citation;
- (done) the figure: Nadal's posterior path over his argmax law,
  nadal.pdf, built by exp6_career_peak/make_figure.py;
- reconcile scope wording with the prior-art audit
  (papers/prior-art-inversion-and-shared-field.md) -- the claims
  section was written to its standard but has not been audited.
No venue chosen.
