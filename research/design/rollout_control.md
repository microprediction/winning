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

## "Search as Computation Allocation" read in full (2026-09-01)
Tuisov, arXiv:2607.27871, AAAI-27 formatting; PDF in refs/. What it
is: the normative scaffolding for exactly our problem class --
terminal computation-allocation with Bellman characterizations for
fixed-budget (Eq. 17), priced (Eq. 19), and certificate-seeking
(Eq. 20) objectives. The engine hooks, now precise:
- Prop. 3: under simple-regret loss, myopic VOC IS the
  knowledge-gradient criterion -- the E[max]-increment our potential
  W differentiates exactly under factor covariance.
- Eqs. 28-29: under zero-one identification loss, myopic VOC is the
  improvement in max_z p(z|H) -- and for best-arm identification
  p(z|H) IS the PoM vector. His Limitations section: "Exact dynamic
  VOC is itself generally intractable." In the factor-Gaussian case
  the myopic quantities are engine outputs.
- Eq. 44 and surroundings (MCTS root VOC): the score "depends on the
  posterior tail that can cross the current decision boundary and on
  the value gap across that boundary. A visitation count alone is
  not sufficient" -- the decision boundary is the tie surface; the
  photo-finish/local-time geometry again. And the UCT approximation
  chain (Eq. 47) explicitly "discards the root value gaps, the
  probability that a local change propagates to the root,
  CORRELATIONS among backed-up values" -- the wedge named by the
  paper itself.
- Prop. 4 / Thm. 2: information gain can rank computations
  arbitrarily poorly against VOC (one-sided bound only) -- caution
  for entropy-style acquisition (relevant to LITE's entropy tasks).
- The certificate-seeking objective is the hard version of our
  exp1_stopping rule (stop at PoM >= 1-delta).
- Conclusion asks for "learned, history-dependent approximations of
  dynamic VOC"; our counter-position: in factor-Gaussian settings
  the myopic VOC needs no learning -- it is exactly computable at
  scale -- and the survivor-set/irreversible-kill control extends
  his computation-set dynamics where killing removes computations.

## Standing request
Track new work on adaptive LLM rollout pruning / compute allocation;
fold anything that derives (rather than tunes) the survivor policy
into this note. Merge with the R&S adjudication (issue 16 agent) when
it reports -- Frazier/OCBA overlap is deliberate.

## The n=2 free boundary: solved (2026-09-02, exp_rollout_n2/)
The conjecture holds and the law is one line. With gap volatility
sigma = sqrt(2(1-rho)) and compute price lambda per path per unit
time, the value relative to the current leader satisfies a
one-dimensional Bellman equation on the reflected gap (the leader's
level is a martingale, so a lone survivor earns nothing; the option
value is the reflection at zero raising the leader). Measured:
- The kill boundary exists, rises with remaining budget, and
  SATURATES: beyond b ~ 0.2 (sigma/lambda)^2 units more budget does
  not widen the keep zone.
- Brownian scaling collapses everything:
  h(b; sigma, lambda) = (sigma^2/lambda) H(b lambda^2 / sigma^2),
  verified numerically (predicted ratio 4.000, measured 4.17 at
  DT = 5e-4; converging in DT). The saturated constant is
  H_inf ~ 0.115 across rho: KILL THE LAGGARD WHEN THE GAP EXCEEDS
  ~0.115 sigma_gap^2 / lambda. Correlation enters only through
  sigma^2 = 2(1-rho): near-duplicate rollouts get pruned almost
  immediately -- the qPO diversity intuition as a control law.
- Adaptivity is worth 2x: at rho=0.5, b=1, the free-boundary policy
  earns 0.0308 (Bellman 0.0310 -- solve certified by simulation);
  the BEST TUNED fixed-gap rule earns 0.0150, the best tuned
  fixed-time rule is negative, keep-always loses 1.60. The static
  schedules the LLM-rollout literature tunes leave half the value of
  the second trajectory on the table even at n=2.
Next: n > 2 via the pairwise-boundary heuristic priced by the
engine's PoM/tie machinery, benchmarked against sequential halving.
