"""The equal-groups table of Section 6.2.

n = 24 contestants in g equal groups with within-group correlation 0.7.
The dense correlation matrix is priced through the shipped one-call
pipeline (race_probabilities(mu, cov=C), which runs fit_covariance) and
compared with the same race priced through the block grammar directly,
which fits nothing. Setup and seed are those of
tests/test_fit_covariance.py::test_group_degeneracy_is_predictable_and_detected.

Printed per g: the multiplicity of the leading eigenvalue of P C P, the
within-group bulk eigenvalue over the leading one, whether the default
rank 3 cuts the tied group, whether the tied-eigenvalue warning fires,
the fitted rank and projected residual from the fit's own report, and
the total variation against the block grammar.

    python equal_groups.py
"""
import warnings

import numpy as np

from winning.factor.core import fit_covariance
from winning.factor.races import race_probabilities
from winning.factor.structures import Blocks

n, rho, k = 24, 0.7, 3
rng = np.random.default_rng(0)
mu = rng.standard_normal(n) * 0.6
mu -= mu.mean()
P = np.eye(n) - np.ones((n, n)) / n

print("groups  mult  bulk/lead  cuts tie  warns  fit rank  proj resid   TV vs grammar")
for g in (2, 3, 4, 6, 8, 12):
    lab = np.repeat(np.arange(g), n // g)
    C = np.where(lab[:, None] == lab[None, :], rho, 0.0)
    np.fill_diagonal(C, 1.0)
    w = np.sort(np.linalg.eigvalsh(P @ C @ P))[::-1]
    mult = int(np.sum(np.abs(w - w[0]) < 1e-8 * abs(w[0])))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p_dense = race_probabilities(mu, cov=C)
        warned = any("tied eigenvalue" in str(x.message) for x in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = fit_covariance(C, return_report=True)[-1]
    blocks = Blocks(cluster=lab, loading=np.full(n, np.sqrt(rho)),
                    D=np.full(n, 1 - rho))
    p_exact = race_probabilities(mu, structure=blocks)
    tv = 0.5 * float(np.abs(p_dense - p_exact).sum())
    print(f"{g:6d}  {mult:4d}  {w[mult] / w[0]:9.3f}  "
          f"{'yes' if mult > k else 'no':>8s}  {'yes' if warned else 'no':>5s}  "
          f"{rep['rank']:8d}  {rep['projected_residual_rel']:.1e}      {tv:.1e}")
