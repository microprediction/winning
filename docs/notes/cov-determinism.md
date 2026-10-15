# Determinism of `race_probabilities(mu, cov=S)` across platforms

Decision record for #745 (2026-10). Related: #688 (fastmvn's
`_canonical_loadings`), #605 and #728 (MNP's Sobol cloud).

## What happened

The default-drift guard pinned `race_probabilities(mu, cov=S)` for an
ordinary 6-runner sample covariance (`tests/ordinary_battery.py`,
`race_cov/sample/n6r3`), and the pinned value differed between macOS and
Linux/Windows by 6e-5 to 1.2e-4.

The first guess was that the route flipped (factor fit vs dense GHK).
It did not. CI on every platform reported the same route: rank 3,
pipeline arm, `degraded=False`. Perturbing S by 1e-15 to 1e-13 never
changed the rank, the floor count, the residual flags or the route
(`tests/test_cov_route_rounding.py`).

The actual cause was the sign of one eigenvector column:

- `fit_covariance` builds V from LAPACK eigendecompositions. On
  Linux/Windows the third column came back negated:
  V[0] = [-1.32791, 0.21495, -0.07010] against +0.07010 on macOS.
- `qmc_nodes` (scrambled Sobol) is not symmetric under a column sign
  flip, so V and -V on one column price the race differently.
- Flipping that column on macOS reproduced the Linux vector to 4.8e-9.
- `scipy.linalg.eigh(driver="evr")` flipped a different column and moved
  the price by 1.06e-4.

## How big that is

On n6r3, against a reference (factor nodes at 2^18 x 8 seeds and dense
GHK at 2^18 x 8 seeds, which agree to 3e-7):

| | max abs error |
|---|---|
| factor default, macOS basis | 8.3e-5 |
| factor default, Linux basis | 6.3e-5 |
| factor default, 16 Sobol seeds | median 6.6e-5, max 1.5e-4 |
| factor default, all 8 column-sign patterns | max 8.3e-5 |
| factor default, canonical basis (this change) | 4.1e-5 |
| dense GHK default (4096) | 2e-6 (max 2.2e-5 over 16 seeds) |

So the platform gap was inside the factor route's ordinary accuracy. It
was a reproducibility defect, not an accuracy defect.

## Options considered

1. **Canonical factor basis (chosen).** Before building nodes,
   `fit_covariance` rotates V into the eigenbasis of V'V (eigenvalues in
   decreasing order). V V' is unchanged up to rounding. Each column's
   sign is then fixed so its largest-|entry| is positive, with ties going
   to the first index. This is `core._canonical_factor_basis`, the same
   construction as fastmvn's `_canonical_loadings` + `_sign_fixed` (#688).
   - Cost: one r x r eigendecomposition.
   - Effect on defaults: only fits that take the factor route move, and
     only by about the QMC error. In the guard battery that was one entry,
     n6r3, by 1.24e-4. The other `race_cov` entries are degraded fits that
     route to GHK.
2. **QMC-sized tolerance in the drift guard.** Accept the platform
   dependence and loosen the guard for `race_cov/*`. Rejected: it hides
   the defect instead of removing it, and a loose guard would also hide
   real regressions.
3. **More nodes for cov=.** This shrinks the gap but does not remove it,
   and it costs time on every call. It addresses accuracy, not
   determinism.
4. **Dense GHK for ordinary cov=.** More accurate on n6r3 (2e-6), but it
   moves every factor-route default, and slopes, tempered races and
   non-normal bases still need the factor form.

## What the canonical basis does not fix

- **Tied or near-tied eigenvalues of V'V.** Any rotation within the tied
  space is equally canonical, and the eigensolver picks one. Pipeline
  fits whose columns are not orthogonal can also round differently once
  the signs are conjugated: prices then agree to about 1e-16, not
  bitwise (`tests/test_cov_canonical_basis.py` allows 1e-14 on that
  battery and requires bitwise equality on n6r3).
- **Ties in the sign rule.** A column whose two largest |entries| are
  equal in magnitude and opposite in sign gets its sign from rounding.
- **The other ports.** R and the browser fit covariances with their own
  node clouds (Halton in R) and are not changed here.

## The deeper alternative

Sign-symmetric (antithetic) node clouds would make the price invariant
to column signs by construction, with no canonicalisation. That covers
signs only, not rotations. Symmetrising has also cost accuracy before:
#605 symmetrised MNP's Sobol cloud, and on Fishing the fit stalled about
3.4 nats short of the plain cloud's optimum (GH truth -1216.30 against
-1212.87). #728 went back to the plain cloud and kept the symmetric
helpers opt-in. Any symmetric cloud for cov= would need the same
measurement first.
