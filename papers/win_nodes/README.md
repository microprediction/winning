# Win Nodes for Factor-Structured Gaussian Bayes Nets

The n-ary win node W = argmax_i X_i on Gaussian Bayes nets whose
variables share a few factors: the forward query, its Jacobian, the
likelihood of an observed winner, a quoted price as evidence, and
published places as evidence, all from the lattice computation of
arXiv:2609.01133; the factor-form posteriors of simulation under common
random numbers and of feature bandits; stopping and bandit experiments.
Merged in 2026-11 with papers/exact_pom. Chain and band parents, which
use a different (classical) recursion, were cut from this paper on
2026-11-09; that material lives in GMRFExtremes.jl and in the git history
of this directory.

Numbers trace to committed scripts: research/rs_crn (exp1_stopping,
exp2_bandit, exp3_quadrature); research/win_nodes/exp1_two_evidence
backs Table 1 (a quoted price as evidence, composed with an observed
winner); research/win_nodes/exp2_place_evidence backs Table 2 (places
as evidence).
arXiv kit: arxiv/.
