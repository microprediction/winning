# Pruning LLM rollouts as stochastic control on a correlated race
(Peter's notes, 2026-09-01. All citations [U] until read at source.)

## What the engineering does today
Tree-of-Thoughts keeps the top b at every depth, b <= 5, externally
chosen. REBASE allocates a fixed expansion budget proportionally to
process-reward scores. ReSCALE (March 2026) runs Sequential Halving --
allocate, compare, eliminate a fraction, repeat: a racing algorithm.
"Thought-Level Beam Search" (Aug 2026; the Gambit algorithm) frames
reasoning as constrained compute allocation over partial trajectories:
warmup, periodic scoring, kill bottom-K, branch top-K at constant
capacity -- but K, the interval and the warmup are tuned, not derived.
The field chooses schedules empirically rather than solving the
control problem.

## The nearby optimal-control literature
Bayesian ranking & selection / optimal learning: knowledge-gradient
(Frazier-Powell-Dayanik), including the CORRELATED-normal case, where
the one-step rule is the expected increment of the maximum from one
more measurement -- an object this engine evaluates natively. OCBA for
budget division; Brownian boundary-crossing elimination procedures,
recently multidimensional. "Search as Computation Allocation" (July
2026) writes the Bellman equations for search-as-terminal-decision;
Rational Metareasoning for LLMs applies value-of-computation to cut
20-37 percent of tokens.

## The gap (the problem worth solving)
Correlated EVOLVING partial trajectories dX = dW (correlated), an
ACTIVE SET A_t as the control, cost |A_t| per unit time, irrevocable
killing, finite budget B, terminal objective max over survivors:

  V_t(x, S, B) = max over nonempty A subset S of
                 E[ V_{t+1}(x_A + dW_A, A, B - |A|) ],
  V_T = max_{i in S} x_i.

The max couples the arms, so no Gittins decomposition; correlation
couples them further. R&S is about learning FIXED qualities; here the
trajectory ITSELF evolves and dies when cut. Existing LLM systems fix
the survivor count; here the survivor-set SIZE is the control.

## Where the winning machinery bites
- The nonsmoothness of max(X) lives on the tie surfaces
  X_i = X_j = max: exactly the photo-finish boundary. The Atlas note's
  Claim 1 makes the intuition computable: the value of keeping two
  nearly tied paths is a local-time object, and the engine prices
  leading-pair collision local-time densities at O(nL) per horizon.
- E[max] is the potential W (min-wins mirror), whose gradient is the
  membership vector: knowledge-gradient increments and marginal
  value-of-compute are derivatives the engine returns exactly.
- Which paths plausibly become the max is the top-k / PoM vector, at
  any horizon, under factor correlation between sibling trajectories
  (shared prefixes ARE shared factors: prefix-incidence loadings, the
  same Gram structure as timing paths).

## The mathematical next step (agreed)
Solve n = 2: two correlated Brownian paths, budget B, keep-both cost
2/unit. Conjecture: the optimal rule is a free boundary
  keep both  iff  |X_1 - X_2| < h(t, B, rho),
kill the loser otherwise -- an optimal-stopping problem where the
local-time connection becomes concrete (Ito-Tanaka on |X_1 - X_2|).
If h exists and is computable, the n > 2 policy has a natural
pairwise-boundary heuristic the engine can evaluate at scale, and the
comparison against tuned beam schedules (ToT/REBASE/Gambit) is the
experiment.

## Standing request
Track new work on adaptive LLM rollout pruning / compute allocation;
fold anything that derives (rather than tunes) the survivor policy
into this note. Merge with the R&S adjudication (issue 16 agent) when
it reports -- Frazier/OCBA overlap is deliberate.
