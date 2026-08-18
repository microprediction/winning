# Experiment 14: the boundary studies

The algorithm paper's honesty sections, run before the claims were written.

**Anchor.** The generic-base factor forward (normal or standardized Gumbel-min
idiosyncratic noise) reproduces softmax *to machine precision* (2.8e-17) with a
Gumbel base and zero loadings — the exact Luce nesting (common-scale condition).

**Standardization note (referee catch, fixed).** The first version of Part B
confounded noise *family* with noise *variance* (raw Gumbel carries π²/6 the
variance; the skew-normal truth had variance 0.427 and nonzero mean). All noise
families are now standardized to mean 0, variance 1 before scaling by √D. The
ordering survived standardization and sharpened.

**Part A — full-covariance boundary** (N=50, dense correlation matrices with
eigenvalue decay λ_m ∝ m^−γ, truth = 8×10⁶-draw MC, GHK at the *exact* Σ as the
reference):

| γ (top-4 eig share) | lattice k=1 → k=8 | GHK R=10³ | GHK R=10⁴ |
|---|---|---|---|
| 0.5 (21%) | 3.8e-3 → 1.3e-3 | 8.0e-3 | 3.2e-3 |
| 1.5 (64%) | 2.4e-2 → 2.6e-3 | 1.3e-2 | 3.4e-3 |
| 3.0 (93%) | 5.6e-2 → 1.2e-3 | 2.2e-2 | 4.1e-3 |

**We expected a clear GHK-wins regime and did not find one at this size**: k=8
matches or beats GHK R=10⁴ at every decay rate. The factor floor decays slowest at
*intermediate* γ (the same mid-spectrum hardness as exp06's kinked kernels), which
is where a GHK advantage would first appear if accuracy demands exceeded the
affordable floor. The boundary is conditional, not territorial — reported as it
fell, with the original expectation corrected in the script docstring.

**Part B — substitution fidelity** (truth misspecified for *every* candidate:
t(5) factors + skew-normal idiosyncratic; candidates calibrated to identical menu
shares with supplied loadings; scored on deletion counterfactuals vs fresh MC).
Blocks whose deleted share sits at the MC noise floor are uninformative (all
models tie, as they must). On informative blocks, TV as a fraction of redistributed
mass:

| model | deleted mass > 10% | deleted mass 2–10% |
|---|---|---|
| plain logit (IIA) | 15.9% | 25.7% |
| factor mixed logit | 7.7% | 17.1% |
| **factor probit** | **2.8%** | **7.8%** |

(Mass-stratified reporting is computed in the committed script itself; blocks
at the MC noise floor are reported as uninformative. Strata are small — 1 block
with mass>10%, 4 with 2–10%, of 24 total — and calibration residuals on menu
shares are 4.9e-11 / 1.1e-9, both printed by the script.)

Factor structure carries the first half of the correction; matching the
idiosyncratic noise *family* carries a further factor of two (caveat: the truth's
skew-normal noise is closer to Gaussian, so read this as "the noise family
matters," not "probit always wins").

Tests: `tests/test_boundaries.py` (softmax anchor, base parity with raceutil,
calibration roundtrips for both bases).

Run: `python run_boundaries.py` (~12 min, numpy/scipy/matplotlib only).
