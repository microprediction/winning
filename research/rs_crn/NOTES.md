# Track D: R&S under CRN + exact Thompson shares
(Adjudicated PURSUE narrow 2026-09-01; report in
../adjudications/rs_thompson.md. Adjacent: ../design/
rollout_control.md shares the KG/OCBA literature.)

## The claims under attack, VERIFIED IN FULL (PDFs in refs/)
- VAPOR (arXiv:2311.13294, read 2026-09-01): p.4 "computing
  [P_Gamma*] involves computing several complicated integrals with
  respect to the posterior and is intractable in most cases"; Sec. 4
  opens "computing this probability is intractable in general".
  Lemma 8 (p.8): E[lambda^TS] = P_Gamma* -- expected TS occupancy IS
  the probability of optimality. p.8: direct access desirable "to
  ensure safety constraints or to allocate budgets"; TS suffers
  linear regret in multi-agent and constrained cases. STRUCTURAL
  FINDING: the VAPOR objective (Eq. 2-3) consumes only marginal
  means and marginal sub-Gaussian widths -- the surrogate is blind
  to posterior correlation by construction.
- ToSFiT (arXiv:2510.13328, ICLR 2026, read 2026-09-01): defines
  PoM(x|data) = P[R_x = R*|data] on the FULL correlated Gaussian
  R ~ N(mu, K), then the VBOS objective (Eq. 1) and the near-closed
  form (Eq. 2, pi = v((mu-kappa*)/sigma), v(c) =
  exp(-(sqrt(c^2+4)-c)^2/8)) use marginals only. On the prior VBOS
  regret bound: "the structure of the kernel is not taken into
  account, i.e., the worst-case bandit with independent arms is
  assumed." Intractability framing: acquisition maximization in
  large discrete domains "becomes intractable."
- CONVERGENCE WITH TRACK A: ToSFiT's authors are Menet, Terzic,
  Hersche, Krause, Rahimi -- Menet and Krause are LITE's authors,
  and ToSFiT Eq. 2 credits "Menet et al. (2025)" (= LITE) for the
  threshold construction. The whole VBOS line (O'Donoghue-Lattimore
  2021 -> VAPOR 2023 -> ToSFiT 2026) plus LITE is ONE research
  program whose every computable object is marginal-only. One short
  paper answers both tracks' counterparties.
- GSP (Ni et al., OR 2017, Sec. 2.5): "our procedure does not
  support the use of common random numbers." [Still to re-verify in
  the PDF before citing.]

## Decisive experiments
R&S: Negoescu-Frazier-Powell drug testbed (literally VV'+D, known
loadings, used by P3C arXiv:2402.02196) -- exact-PoM stopping and
allocation vs KN, P3C-KN, OCBA at matched PCS; report sample-size
ratio. Thompson: finite correlated Gaussian bandit -- exact-PoM
sampling vs VAPOR/VBOS vs TS on regret AND on VAPOR's own
approximation-gap metric. Synthetic data: no acquisition risk.

## Status
OPENED 2026-09-01 on the Thompson side. exp1_stopping/ runs the
stopping experiment: R&S under CRN, posterior exactly
factor-plus-diagonal at every step (Woodbury on the CRN precision --
derivation in the script docstring), three rules stopping on their
own max PoM >= 0.95 (exact engine race; F-LITE independence
construction; VBOS Eq. 2), in aligned / opposed / independent factor
regimes. Engine certified against 200k-sample MC (TV 0.0017); at
n=20 aligned the marginal-only estimates are 7-8% TV from exact.
Remaining for the short paper: the bandit regret companion
(exact-PoM sampling vs VBOS vs TS -- Lemma 8 says exact-PoM sampling
IS expected TS, so the interesting exhibits are the gap metric and
derandomized/stopping uses), and the Negoescu-Frazier-Powell drug
testbed for the R&S allocation half.
