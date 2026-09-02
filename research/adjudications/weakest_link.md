# Adjudication: weakest-link crossarms (Track C)
(Agent report, 2026-09-01. Verdict: PURSUE at one-experiment scale;
do not over-invest.)

## Their model, precisely (arXiv:2608.01261, He & Wong, read in full)
Lognormal strength (Eq. 3): log y_ij = mu + X_ij' beta + theta_i,
zone covariates X (knot area/count), theta_i ~ N(0, sigma_theta^2)
specimen effect. NO ZONE-LEVEL NOISE -- y_ij deterministic given
(mu, beta, theta_i). Stress profile log psi_j = sum_g B_jg gamma_g,
B-splines, G=5, sum-to-zero, physics-informed prior. Surrogate
(Eq. 7): Softmin weights w_ij = exp(-k log z_ij)/sum, k FIXED
(claimed non-identifiable), k=5 by 10-fold CV (Brier minimal at 5,
worse k>=6). Failure load y_obs | f_i ~ N(log y_{i,f_i},
sigma_eps^2), sigma_eps = 0.01 fixed. Stan/NUTS, 4x10,000.
Robustness: k in [1,10] sweep; knot-coefficient sign/CI stable,
"minor deviations in mu and sigma_theta," Brier k-sensitive. No
comparison to exact argmin or any alternative surrogate; no
Luce/Gumbel-max framing (their Softmin IS a Gumbel race on log z
with scale 1/k -- unrecognized).

## The honest core
r = 5 zones, conditionally INDEPENDENT given theta_i (spatially
correlated knot clusters explicitly their future-work limitation).
theta_i cancels inside the argmin, so with no zone noise the true
argmin is DETERMINISTIC; the Softmin's k smuggles in the missing
randomness. The principled fix: zone-level error e_ij ~ N(0,
sigma_e^2) in log-strength, making the failure zone an exact 5-horse
Gaussian race -- and sigma_e REPLACES 1/k and IS IDENTIFIABLE (the
observed failure load pins the scale of log z; their
non-identifiability argument holds only at zero zone noise). Kills
the fixed-k CV step entirely.

BUT: for the (zone, load) joint likelihood under independence,
exactness is elementary -- phi_j(t) prod_{m!=j} Sbar_m(t), five
lines of Stan. winning is decisive only for (a) marginal zone
probabilities (load integrated out; 1-D quadrature suffices at n=5
independent), and (b) CORRELATED zone strengths -- their stated
limitation, where the product form dies and factor/block covariance
with exact gradients has no cheap substitute.

## Kill risks / data
- Data PUBLIC: Anderson 2019 MS thesis, ScholarsArchive@OSU, id
  zc77sw48x (bot-gated to fetch; machine-readability UNVERIFIED).
  Paper reports 198 specimens -- the "200 rejected + 50 accepted"
  framing in our notes DID NOT VERIFY; no censored/accepted
  specimens appear in the paper. No code release found.
- n=5 independent race: exactness per se too cheap to be the pitch
  (the strongest kill risk).
- Softmin partially fine in practice: coefficient signs robust; the
  wedge (Brier, mu/sigma_theta drift) exists but is not huge.

## Decisive experiment
Refit on the public data with lognormal zone noise + exact race
likelihood; report (i) posterior for sigma_e (identifiability
demonstrated), (ii) held-out failure-zone log loss / Brier vs their
k=5, (iii) stress-profile gamma distortion vs k. Then add a
knot-cluster spatial factor. Contact He & Wong (Waterloo) only after
(i)-(ii) succeed.

## Positioning
Not "exact beats surrogate" -- "identifiable zone noise replaces the
artificial k, and correlated knot clusters, their stated limitation,
become tractable."
