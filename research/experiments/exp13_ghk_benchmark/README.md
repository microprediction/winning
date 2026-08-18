# Experiment 13: the GHK benchmark

**The claim under test** (paper/algorithm-target.md): the lattice fast ability
transform with factor conditioning computes multinomial-probit choice probabilities
for all N alternatives deterministically, at scales where the GHK simulator — the
standard tool of discrete-choice econometrics — is unusable; and it inverts market
shares to utilities (the probit BLP step), which GHK-based practice essentially never
attempts.

**Anchors first** (nothing was compared before both methods passed ground truth):
N=2 closed form — lattice 3.3e-16, GHK 2.4e-14 (GHK implementation validated);
N=5 vs 10⁷-draw MC — both within noise; package parity — `thurstone.FactorRace`
agrees with the benchmark implementation of the same algorithm to 1.1e-5, so the
claims attach to the shipped library.

**Full share vector: accuracy and time** (k=2 factors; MC truth 2×10⁶ draws):

| N | lattice | GHK (R=1000) |
|---|---|---|
| 5 | 20 ms, err 4.1e-4 | 1 ms, err 3.1e-3 |
| 20 | 102 ms, err 3.8e-4 | 13 ms, err 4.4e-3 |
| 50 | 247 ms, err 2.1e-4 | 76 ms, err 7.7e-3 |
| 200 | 870 ms, err 2.8e-4 | 1,277 ms, err 6.7e-3 |
| 1,000 | 4.3 s, err 8.0e-4* | — |
| 5,000 | **22 s, err 9.0e-4*** | — |

\*at the MC-truth noise floor (~4e-4). Lattice error is flat in N; GHK error grows.

**The matched-accuracy comparison (the honest headline).** GHK's error scales as
R^(−1/2). Extrapolating its cost by the *measured* power law (α ≈ 2.0 over the tested
range — a lower bound; the asymptotic per-share cost is cubic), GHK at N=5000 and
R=1000 would take ≥ ~13 minutes and deliver ~7e-3 accuracy; matching the lattice's
9e-4 requires R ≈ 55,000, i.e. **≥ 12 hours against 22 seconds** — three orders of
magnitude, using extrapolation assumptions that favor GHK. (The earlier in-script
extrapolation used a naive N³ constant; corrected here from the empirical fit.)

**Derivative smoothness** (curvature noise of P(μ+t·e) along a line; the
estimation-relevant metric): lattice 0.01, GHK-CRN 0.01, GHK-fresh 39.3. The nuance
matters: common random numbers make GHK smooth conditional on its draw set;
its curve sits off the lattice curve (figures/smoothness.png), a discrepancy
between two approximations that only an independent high-accuracy reference
could decompose. The correct claim is "both methods differentiate their own
approximations;
the lattice needs no trade."

**Share inversion (probit BLP step), N=1000, k=2.** Target shares from 5×10⁶ MC
draws (no inverse crime). **59 seconds**; forward-match 6.3e-9 — four orders of
magnitude below the target's own noise — and utility recovery to 0.014 on
identified alternatives. The inverter is the original fast-ability-transform
design generalized: coordinate-Newton against a frozen field with analytic
own-ability slopes from the same lattice pass, the independent-inverse warm start
of the allocation package, and a tail-aware tolerance (an earlier unaccelerated
Picard loop took 63 minutes; the redesign is 64×). Further headroom recorded: the
original algorithm's interpolation trick generalizes (per-node ability→probability
curves are cross-correlations, computable at all offsets at once).

**Assortment ensemble, N=200**: every single-removal share vector from one
conditional field pass, 3.8× faster than recomputation, identical to 5.6e-17.

**Verdict for the paper.** All four pillars hold on measured evidence: (1) flat-error
deterministic shares where GHK degrades and then becomes infeasible; (2)
resampling-free derivatives of a fixed-design approximation (not exact MNP
derivatives — both methods differentiate approximations); (3) share inversion
at N=1000 in under a minute, matching
targets to below their own noise; (4) the one-pass assortment ensemble. Remaining
before submission: the full-covariance (non-factor) boundary study where GHK
retains the advantage of arbitrary Σ, and the mixed-logit substitution comparison.

Tests: `tests/test_ghk_benchmark.py` (closed-form anchor for all three methods, MC
agreement, CRN determinism, normalization, inversion roundtrip).

Run: `python run_ghk_benchmark.py` (~15 min; numpy/scipy/matplotlib only).
