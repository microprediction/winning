# Experiment 15: the perturbation certificate — what survived verification

Goal: an a-priori error bound for factor-approximated probit (the paper's
would-be missing lemma). Status after testing, per house rules:

**Verified.** The first-order derivative identity: for i ∉ {j,k},
∂p_i/∂Σ_jk = t_ijk, the **triple-tie density** ∫ f_i f_j f_k ∏ S_l dx
(Gaussian integration by parts / Price's theorem) — matches finite differences
to 7.9e-6 with the sum rule Σ_m ∂p_m/∂Σ_jk = 0 exact to 1e-14. All pair totals
T_jk come from one extra O(N²L)-per-node field pass via hazard products.

**Refuted (both, by test).** (1) Slepian-style negativity of winner-involving
derivatives: a strong alternative can *gain* from correlating with a weak rival
(+1.2e-2 observed). (2) The bounding conjecture |∂p_i/∂Σ_jk| ≤ T_jk: violated
at ratio up to **1.27**. So the clean theorem does not hold as conjectured.

**What remains true and useful.** The certificate Σ_{j<k}|ΔΣ_jk|·T_jk held in
100% of tests (synthetic factor residuals and real dense-Σ-minus-rank-k
residuals), running 6–30× conservative — an empirical conservative estimate,
not a theorem. Linearity of error in the residual: log-log slope 0.97–0.98
across two decades, three norms.

**Numerical trap fixed en route.** `1 − ndtr(z)` underflows to 0 at z ≳ 8,
exploding hazards f/S to 1e150; use `log_ndtr(−z)` for exact log-survival
(hazards are then Mills-ratio bounded).

Tests: `tests/test_certificate.py`. Run: `python run_certificate.py` (~3 min).
