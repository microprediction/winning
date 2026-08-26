# PQ search as a race: calibrated per-query rerank depth

Product-quantized search reranks the ADC top-m exactly; m is a hand-tuned
global constant in every production ANN system. "Is the true nearest
neighbour in my shortlist?" is a place question in a race over N candidates
with noisy abilities, so the right m is per-query: smallest shortlist whose
summed win probability reaches the target (the events are disjoint, so
coverage is additive -- qPO's identity, used for good this time).

Data: QM9 fingerprints (133,885) random-projected to 128d, unit-normalised;
PQ 16 blocks x 8 dims x 256 codes; 500 held-out queries; exact brute-force
ground truth.

## Three versions, each a diagnosis

**v1** treated the fitted per-(block, code) error variance as independent
noise: asked 95%, delivered 100% at mean depth 289 against a fixed depth of
32 giving 98.2%. Useless. Diagnosis: most of that variance is COMMON to all
candidates (query geometry) or shared within codebook cells, and a shift
that moves every score together cannot change the argmin. The racing
common-mode lesson, in costume.

**v2** projected out the query-common and code-shared components: barely
moved (266 at 95%). Second diagnosis: the variance was fitted on random
(query, vector) pairs, i.e. FAR pairs, whose error geometry is nothing like
the near-neighbour region where races are decided.

**v3** uses the algebra instead of estimation. The ADC error is exactly
e_ib = (||x_b||^2 - ||c_b||^2) - 2<q_b, r_ib>: a per-candidate constant
known at index time, plus a zero-mean query-linear term whose variance is
proportional to the stored residual energy rho_i. So store TWO extra floats
per vector (const_i, rho_i) next to the 16 code bytes: exact debias, and
per-candidate race variance v_i = kappa rho_i with kappa calibrated once
(measured 0.038 vs the isotropic prediction 0.031).

## v3 result: adaptive beats fixed at matched coverage

    target   mean m   p90    max   achieved coverage
      0.90     58     117    305       0.992
      0.95    107     214    538       0.998
      0.99    311     610   1330       0.998

Fixed-depth coverage on the same (debiased) ordering: m=32: 0.942,
m=64: 0.976 -- reaching the adaptive rule's 0.992 needs a fixed m well above
100. So at matched coverage the adaptive rule spends roughly HALF the
average rerank budget, while carrying a per-query statement instead of an
empirical average. It is conservative (over-covers its nominal target),
which for a coverage guarantee is the safe direction.

## Two honest surprises

1. **Exact debiasing made the RANKING worse.** The oracle fixed depth under
   exactly-debiased scores is 21/43/140 (targets .90/.95/.99) against 9/14/40
   under v2's biased scores. Unbiased per-candidate estimates are not the
   best ranker: the biased ADC score is a shrunk estimator, and shrinkage
   helps ordering. (The race framework accommodates this -- rank by win
   probability, which shrinks automatically -- but the interplay deserves its
   own experiment.)
2. **The race overhead is currently 42 ms/query** (a 129-point quadrature
   over 4,096 candidates in numpy) against reranks that cost microseconds
   each. The DEPTHS are won; the arithmetic must get ~100x cheaper (fewer
   candidates in the race, fewer points, or the closed-form Gumbel-style
   approximation) before the wall-clock trade is interesting. This is an
   engineering gap, not a conceptual one, and it is the next task.

## Files

    run_pq_race.py    v1 (independent fitted variance)
    run_pq_race2.py   v2 (common-mode and code-shared projected out)
    run_pq_race3.py   v3 (exact bias + residual-energy variance)
    log_pq_race*.txt  outputs
