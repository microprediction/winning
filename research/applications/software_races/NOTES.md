# Abstract races-to-fail in software (the literal correction)
(2026-09-03. I was being literal about physical failure and hit a
data wall; the races-to-fail in software are abstract, their
correlation is the SOURCE CODE, and the ground truth SIMULATES. All
references [U] until read; prior-art agent out.)

## The reframe
A "race to fail" in software is not a disk dying. It is:
- TEST SUITE catches a bug = P(at least one of k tests fails | bug)
  = 1 - P(all pass), an extremal race; tests sharing coverage FAIL
  TOGETHER, and the coverage matrix IS the factor loading. This is
  pass@k (eval-stats thread) wearing a testing hat.
- COMPILE/BUILD fails at the first broken unit; a broken shared
  header fails every dependent -- a min-race over a correlated
  dependency DAG read off the imports.
- FUZZING/seed selection: which inputs trigger the first crash;
  inputs on the same path are redundant (correlated coverage).
- N-VERSION software: Knight-Leveson 1986 -- independent versions
  fail on CORRELATED inputs (hard inputs hard for everyone), so
  redundancy underperforms independence. The Eckhardt-Lee model is a
  latent-factor "difficulty function" = a factor-probit race. Old
  and directly relevant; check whether anyone used it for SELECTION.
None is Ford (hardware/racks). The correlation is OBSERVABLE
(coverage, imports, paths), so no data wall: simulate the ground
truth from structure, or use real benchmarks (Defects4J: real bugs +
triggering-test matrices; coverage.py matrices for any project).

## exp: correlated test selection, simulated (run_test_selection.py)
800 tests, 200 modules, heavy-tailed module popularity, coverage as
the factor loading; a random bug in ~2% of modules; a test catches
it if it covers a buggy module (detect 0.8). Select k tests to
maximize catch probability, scored by fault-detection over 40k random
bug placements (Monte Carlo ground truth):
  k=5:  correlated 0.452 vs independent-top-k 0.398  (+14%)
  k=10: 0.650 vs 0.591  (+10%)
  k=20: 0.806 vs 0.767  (+5%)
  k=40: 0.917 vs 0.862  (+6%)
The correlated rule covers more DISTINCT modules (37 vs 29 at k=5,
144 vs 110 at k=40): it spreads coverage instead of piling redundant
tests onto popular modules -- the duplicates-vs-specialist result in
a testing domain. Biggest lift at SMALL budgets, which is the
fail-fast CI regime that matters. Fully simulated; the "we need
public failure data" objection that sank Backblaze does not apply.

## The winning connections
- catch probability = the group extremal-value object
  (research/selection, research/cavity_calculus).
- coverage matrix = factor loading V; a latent bug factor F over
  modules; test i fails if it covers a buggy module = a
  factor-conditional Bernoulli, exactly the pass@k factor structure.
- selection = select_race_group (submodular, priced from one field).
- "which test/module is the weakest link" = criticality; "drop this
  redundant test" / "add one covering module m" = the removal/
  addition counterfactual.

## Honest flags (pending the agent)
"Additional greedy coverage" (avoid redundant coverage) is a known
test-prioritization heuristic; my correlated greedy may be a
probabilistic version of it. The likely-novel piece is the UNIFICATION
-- exact P(catch) under a factor-probit coverage model, the same
engine as pass@k, with selection/criticality/counterfactual from one
field -- not the greedy step alone. Eckhardt-Lee already has the
latent-difficulty factor for RELIABILITY; the SELECTION use may be
open. Await the prior-art verdict before any claim.

## Next
Swap the synthetic coverage for a REAL matrix (Defects4J or a
coverage.py dump of an open-source suite) and show correlated
selection beats independent top-k on real bugs. This is the
demonstrable, data-available, not-Ford software fit -- the answer to
"we can simulate".
