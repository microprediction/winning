# select_race_group: group-entry optimization
(Peter's API suggestion, 2026-09-01; prototype verified same day in
exp1_group/. Tracked as a package issue.)

## Suggested API (Peter, verbatim shape)
    result = select_race_group(
        mu=mu, costs=costs, budget=budget, max_size=max_size,
        outside_score=outside, V=V, D=D, base="normal",
        method="auto",   # prefix, greedy, branch_bound, or micp
    )

## Semantics adopted (stated because the suggestion carried no
## objective spec)
Choose a group S of entrants -- unselected candidates DO NOT run --
maximizing f(S) = P(min_{i in S} X_i < X_out) subject to
sum_i costs_i <= budget and |S| <= max_size, min-wins, X_out the
outside competitor summarized by outside_score (scalar threshold or
(mu, sigma, loading) triple). The OTHER reading -- everyone runs and
the group must contain the winner -- makes the objective modular
(sum of fixed win probabilities), a plain knapsack needing no new
machinery; that reading gets a one-line helper at most.

## Structure (why the methods are what they are)
- f is monotone SUBMODULAR: conditional on (z, t), group-miss is
  prod_{i in S} S_i(t|z) with per-member factors in [0,1], so
  Delta_i(S) = E[f_out prod_S S_j (1 - S_i)] is pointwise dominated
  as S grows; expectation preserves it. One line, and verified
  numerically (500 nested trials, zero violations).
- The cavity trick prices ALL n marginal gains in one field pass per
  greedy iteration: W = f_out prod_S S_j once, then Delta_i =
  int W (1 - S_i) for every i. Greedy is O(k n L Q) total.
- Greedy therefore carries (1 - 1/e) under cardinality; under a
  knapsack budget, cost-ratio greedy + best-feasible-singleton
  safeguard carries the standard (1 - 1/e)/2-type guarantee.
- prefix (sort by mu, take best feasible prefix) is the natural
  baseline and is what correlation breaks -- see the cluster demo.
- branch_bound: exact for small n using the submodular upper bound
  f(S) + sum of top remaining gains. micp: deferred -- unclear it
  beats branch & bound at the sizes where exactness matters.

## Measured (exp1_group/results.json)
- f(S) vs 2M-sample MC: max error 6e-4 (MC noise level).
- Submodularity: zero violations, 500 nested trials.
- n=12 knapsack referee: greedy = prefix = brute-force optimum
  (ratio 1.0000 on this instance).
- THE HEADLINE: two tight opposed clusters, pick 4 -- greedy splits
  2+2 for f = 0.591; the mu-sorted prefix piles into one cluster for
  f = 0.393. Correlation-aware selection is worth +50% relative here,
  and no marginal-only method can see it (both clusters have
  identical marginals to within 0.05 in mu).
- n=5000, pick 10 under budget: 9.4s end to end.

## Package plan
winning/factor/select.py: select_race_group with methods auto
(brute for n<=14 else greedy), prefix, greedy, branch_bound; micp
deferred. Result object: indices, value, spent, per-step gains, and
the marginal-gain vector at the optimum (the "who would we add with
more budget" diagnostic). Rust port after the Python surface
settles. Connects to: qPO batch diversity (same diminishing-returns
mechanism), Track E router failover (removal is the mirror of
addition), and the D-optimal scheduler negative (this objective is
the one where uncertainty targeting DID tie -- selection here is a
different question and measurably non-trivial).
