# Adjudication: R&S under CRN + Gaussian Thompson (Track D)
(Agent report, 2026-09-01. Verdicts: R&S PURSUE narrow;
Thompson PURSUE as evaluator, park if derandomized-TS test fails.)

## Incumbent practice
Parallel R&S: PyPRS (Huang, Jiang & Zhong, INFORMS J. Computing,
doi 10.1287/ijoc.2024.1045; github.com/simulation-optimization/
PyPRS; Ray-based). Ships GSP, PASS, KT, FBKT -- all elimination/
tournament designs targeting frequentist PCS/PGS. Flagship scale:
GSP (Ni-Ciocan-Henderson-Hunter, OR 2017) at 1,016,127 systems.
The "16,384 alternatives" figure did not surface -- UNVERIFIED.

CRN handling: GSP states flatly "our procedure does not support the
use of common random numbers" (parallel wing moved to independent
streams for validity). KN/KN++ absorb CRN through pairwise
difference variances; Rinott and OCBA assume independence; OCBA-CRN
(Fu-Hu-Chen-Xiong, IJOC 2007) maximizes an APPROXIMATE PCS from
pairwise terms; Gorder-Kolonko (arXiv:1410.6782) models full unknown
covariance but approximates the posterior. Newest counter-trend:
P3C "Clustering and Conquer" (arXiv:2402.02196) treats CRN
correlation as an asset -- but only to CLUSTER; selection is still
KN/KT elimination. Its flagship testbed (Negoescu-Frazier-Powell
drug discovery, ~1e5 Free-Wilson drugs) is LITERALLY a linear factor
model with known loadings -- VV'+D -- and no exact PoM is ever
computed on it.

CLAIM (b) VERDICT: ACCURATE. Across KN, Rinott, GSP, KT/FBKT, OCBA
(APCS Bonferroni bounds), Chick-Inoue EOC, Frazier's correlated KG
(exact only for one-step KG line integrals; posterior PCS itself
Monte-Carloed): no procedure evaluates the exact joint PoM vector
under a correlated Gaussian posterior. Bonferroni/Slepian bounds,
pairwise screening, or posterior sampling substitute uniformly.

## Correspondence
Old and strong: Nelson-Matejcik (1995) justify CRN selection under
SPHERICITY -- Var(Xi - Xj) = 2 tau^2 for all pairs -- compound
symmetry, i.e. rank-1 + diagonal. Claim (a) is its generalization;
the field's own benchmark already IS VV'+D. VAPOR
(Tarbouriech-Lattimore-O'Donoghue, NeurIPS 2023, arXiv:2311.13294)
and ToSFiT (IBM, ICLR 2026, arXiv:2510.13328) both name posterior
probability-of-optimality, declare it intractable ("several
complicated integrals"), and build variational surrogates (ToSFiT
fine-tunes an LLM toward the VBOS policy under a Gaussian/GP
posterior). An exact evaluator plugs in as (i) the exact fine-tuning
target in ToSFiT's Gaussian layer, (ii) ground truth for VAPOR's
bandit-case approximation, (iii) VAPOR's own motivating uses (safety
constraints, budget allocation).

## Leverage, conditional
Under CRN the estimator covariance is general PSD; low-rank+diagonal
is a MODEL, exact when scenario effects enter near-linearly. Breaks:
nonlinear scenario response leaves residual correlation outside D;
loadings estimated from n0 replications (for k >> n0 factor analysis
is the only feasible estimator anyway -- a point in favor); PoM
inherits estimation error with no frequentist guarantee. Wins:
Bayesian-branch quantities currently bounded or sampled -- posterior
PCS stopping rules, exact OCBA-style allocation gradients via the
Jacobian, O(nLQ) at 1e5-1e6 where MC-PoM is hopeless, and removal
counterfactuals (re-price after elimination; no incumbent analogue).

## Kill risks
1. CURRENCY MISMATCH (serious): mainstream guarantees are worst-case
   frequentist PCS; exact Bayesian PoM answers a question the KN/KT
   school does not ask. Customers are the Bayesian branch (Chick,
   Frazier, OCBA) and stopping rules -- a narrower beachhead.
2. Parallel practice abandoned CRN (P3C is the counter-trend).
3. Pure Thompson needs one draw, not the vector. The vector is
   needed for: derandomized/expected TS (exact PoM IS E[TS
   occupancy], VAPOR Lemma 8), top-two TS, IDS, argmax-entropy
   stopping, budget allocation, constrained/multi-agent settings
   where TS provably fails (VAPOR Sec. 7). Caveat: VBOS/VAPOR's
   optimistic tilt is DELIBERATELY not PoM (their regret proof uses
   it) -- "exact beats variational" must be argued empirically.

## Decisive experiments
R&S: Negoescu-Frazier-Powell drug testbed (already factor-form, in
P3C's paper) -- exact-PoM stopping/allocation vs KN, P3C-KN, OCBA at
matched PCS; report sample-size ratio.
Thompson: finite correlated Gaussian bandit -- exact-PoM sampling vs
VAPOR/VBOS vs TS on regret and on VAPOR's own approximation-gap
metric.

## Positioning
"Under CRN, the covariance R&S procedures bound around is a factor
model; winning prices the whole race exactly, at the scale of GSP."
"The quantity VAPOR and ToSFiT declare intractable is exact and
O(nLQ) in the factor-Gaussian case."

## Locators
PyPRS doi 10.1287/ijoc.2024.1045; GSP people.orie.cornell.edu/shane/
pubs/ParallelRS.pdf (CRN quote Sec. 2.5); P3C arXiv:2402.02196;
OCBA-CRN IJOC 19(1):101-111; VAPOR arXiv:2311.13294 (PDF saved in
session tool-results); ToSFiT arXiv:2510.13328. Unverified: the
16,384 study; Nelson-Matejcik text (existence confirmed via GSP
bibliography, not opened).
