# Order statistics of large factor-correlated populations
(Peter's notes, 2026-09-01. The opportunity is NOT "order statistics
are hard" -- independent non-identical computation is classical
(permanents, recurrences). The vulnerable region is heterogeneous +
correlated + named ranks/spacings + large n, where conditional
factorization collapses otherwise-multivariate calculations. All
literature signals below [U] until read at source.)

## The master object
Conditional on factor node q, the two-state polynomial

    C_{q,x}(z) = prod_j [S_{jq}(x) + z F_{jq}(x)],

whose z^m coefficient is P(exactly m below x | q). With the cavity
polynomial C^{(-i)} excluding i,

    P(R_i = r) = sum_q w_q int f_{iq}(x) [z^{r-1}] C^{(-i)}_{q,x} dx.

## The six targets, ranked
1. WINNER-RUNNER-UP GAP (implement first; tiny code change).
   i wins by more than g iff i at x and everyone else above x+g:
     P(D_1 > g, I = i) = int f_i(x) G_g(x)/S_i(x+g) dx,
     G_g(x) = prod_j S_j(x+g)  -- the survival field SHIFTED by g.
     P(D_1 > g) = int G_g(x) sum_i f_i(x)/S_i(x+g) dx.
   All n winner-specific margin laws from one field; factor case
   nodewise. Literature: normal spacings called harder than plain
   order statistics even iid; a 2021 open question asks for the
   top-two gap law of a general correlated Gaussian vector. We give
   the numerical law for factor-correlated vectors at huge n.
2. k-TH SPACING. D_k > g iff exactly k-1 below x, none in [x, x+g],
   rest above x+g:
     P(D_k > g) = sum_i int f_i(x)
                  [z^{k-1}] prod_{j != i} [S_j(x+g) + z F_j(x)] dx
   -- the rank polynomial with its two states separated spatially
   (below x vs above x+g). O(nkLQ) for fixed k. Possibly a cleaner
   theorem than the rank extension itself.
3. RANGE. Unique minimum i at x, all others in (x, x+r]:
     P(R <= r) = int B(x;r) sum_i f_i(x)/B_i(x;r) dx,
     B_i(x;r) = F_i(x+r) - F_i(x),  B = prod B_i.
   Against Gupta-Pillai-Steck (Biometrika 1964, "...with emphasis on
   range"): general correlation solved only for n = 2,3,4;
   equicorrelation the tractable case. The factor grammar is far
   richer than equicorrelation and scales to enormous n.
4. NAMED-RANK MATRIX P_ir = P(X_i = X_(r)), r <= K: the n x K
   rankogram at O(nKLQ). THE CORE COMPUTATION ALREADY SHIPS:
   winning/factor/topk.py rank_probabilities (count DP +
   stable-direction deconvolution, tested in tests/test_topk.py) --
   what remains is positioning and a demo, not code. Exactly what
   network meta-analysis wants
   (rank probabilities currently from posterior draws/resampling);
   NMA is small-n but validates the output as wanted. Do NOT claim
   novelty for independent k-th order statistics -- permanent
   recurrences already serve that.
5. DEPENDENT k-OUT-OF-n RELIABILITY (strongest applied paper). An
   Aug-2026 paper calls dependent heterogeneous systems "relatively
   underdeveloped" (copulas + ranking-pattern probabilities). Under
   a Gaussian factor COPULA the conditional failure indicators are
   independent Bernoullis and our polynomial is the conditional
   failure-count pgf: P(T_(k) <= t) = P(N_t >= k), plus "who fails
   k-th", plus top-K failure ranks -- with arbitrary positive
   lifetime marginals (copula construction, not Gaussian times).
6. WINNER-TO-RUNNER-UP OPERATOR (labeled inference, prior art
   unestablished). A_ij = P(i first, j second) =
   int G(y) o_i(y) h_j(y) dy with o = F/S, h = f/S: an integral of
   rank-one matrices. Materializing costs O(n^2) (output size) but
   Av costs O(nLQ) matrix-free -- the analogue of the photo-finish
   Laplacian Jv. "Who tends to follow whom in rank" as an enormous
   graph without constructing edges.

## The recursive order cavity (Peter, same date: "I overstated the
## obstruction")
A SPECIFIED ordering is one-dimensional after a recursion. Min-wins,
conditional on a factor node, for the event
pi_1 < pi_2 < ... < pi_k < everyone else:

    H_1(x) = F_{pi_1}(x),
    H_m(x) = int_{-inf}^x f_{pi_m}(t) H_{m-1}(t) dt,

so H_m(x) = P(X_{pi_1} < ... < X_{pi_m} < x), and

    P(top-k prefix in that order)
      = int f_{pi_k}(x) H_{k-1}(x) prod_{j not in pi} S_j(x) dx,

the outside product being the usual survival cavity G/(prod_m
S_{pi_m}). With k = n the outsiders vanish and ONE SPECIFIED FULL
PERMUTATION is H_n(inf): O(nL) independent, O(nLQ) factor -- a chain
of cumulative one-dimensional integrals in place of an
(n-1)-dimensional orthant integral. k=2 recovers
A_ij = int f_j F_i prod S_l; k=3 needs one inner H.

IMPLICATION FOR RANK-DATA LIKELIHOODS: an observed ranking (full or
top-k with ties to "rest") is ONE permutation/prefix -- so a
correlated Thurstone / factor-Gaussian rank likelihood is evaluated
exactly by these recursions, no MVN orthant integral, no simulation.
Next literature search: exact top-k Thurstonian rank-order
likelihood under factor-correlated errors -- this may knock down
something still treated as a high-dimensional Gaussian integral.
Numerical wrinkle: the winner-concentrated lattice window is wrong
for deep recursions; a full permutation needs the window to follow
the order to the last finisher (top 2-5 only needs the low order
statistics' bulk, which the count field locates).

## The sharp boundary, corrected (candidate organizing theorem)
Field-friendly: counts below a threshold; who has rank r; fixed-k
spacings and range; ANY specified permutation or top-k prefix (the
recursion above). Still hard: joint VALUES of several order
statistics P(X_(k1) <= x1, ..., X_(kd) <= xd) -- the conditional
count state across d+1 intervals grows combinatorially in d (a 2021
DP paper calls even the independent case difficult) -- and
ENUMERATING all top-k tuples, whose output size n(n-1)...(n-k+1) no
algorithm beats. Queried orderings are cheap; enumerated order
statistics' joint laws are not.

## Paper candidate
"Order Statistics of Large Factor-Correlated Populations by
Shared-Field Quadrature": spacings + range lead (the surprise),
named ranks second, reliability the applied companion.

## First experiment
exp4_spacings/: winner margin P(D_1 > g) and range P(R <= r) under
rank-1 factor + diagonal; referees = independent-case direct
quadrature and small-n arbitrary-covariance Monte Carlo; then the
n-scaling measurement.
