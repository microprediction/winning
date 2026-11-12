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

Numbers trace to committed scripts and outputs: research/rs_crn
(exp1_stopping, exp2_bandit, exp3_quadrature); research/win_nodes/exp1_two_evidence
backs Table 1 (a quoted price as evidence, composed with an observed
winner); research/win_nodes/exp2_place_evidence backs Table 2 (places
as evidence).
arXiv kit: arxiv/.

Provenance: research/rs_crn/exp1_stopping/results.json and
exp2_bandit/results.json were regenerated on 2026-11-11 from a clean
checkout of d072375b (no uncommitted changes; Python 3.12.13, numpy 2.5.4,
scipy 1.18.1; explicit integer seeds per regime, recorded in each file).
Every number agrees with the earlier run (from an uncommitted runner at
81d6402) to within 1e-9 relative; only provenance fields changed. Each
run's raw rows are in the replications_<run_id>_<pid>.jsonl beside its
results.json. Rerun: `cd research/rs_crn/exp1_stopping && python
run_stopping.py` (about 30 minutes on a 28-core machine) and likewise
`exp2_bandit/run_bandit.py` (about 10 minutes).
