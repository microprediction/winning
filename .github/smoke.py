"""Post-publish smoke test: exercise the winning wheel installed from PyPI.

Run as ``python .github/smoke.py`` so that ``sys.path[0]`` is ``.github/``
(which has no ``winning`` package) -- i.e. ``import winning`` resolves to the
*installed* wheel, never the source tree. The assertion below makes that
guarantee explicit.

What it checks is what a first-time user touches in their first ten minutes:
the front door prices a correlated race, the inverse comes back, the Gumbel
base really is softmax, the classic API is importable at its new home and
its old one, and the ratings layer loads. Plus the shipped contract, which
is the package's own claim about itself.
"""

import os
import warnings

import numpy as np

import winning

ws = os.environ.get("GITHUB_WORKSPACE", "")
assert not winning.__file__.startswith(os.path.join(ws, "winning")), (
    f"imported the source tree, not the installed wheel: {winning.__file__}"
)
print("version:", winning.__version__, "| from", winning.__file__)

# the package's own contract: goldens, identities, round trips
import winning.contract as contract

assert contract.verify(), "winning.contract.verify() failed on the wheel"

# front door: a correlated race prices in one call
n = 6
mu = np.linspace(-0.5, 0.5, n)
V = np.full((n, 1), 0.45)
D = np.full(n, 0.6)
p = winning.race_probabilities(mu, V=V, D=D)
assert p.shape == (n,), p.shape
assert abs(float(p.sum()) - 1.0) < 1e-10, float(p.sum())
assert np.all(p > 0)

# and inverts
from winning.factor.races import abilities_from_race, softmax_probabilities

mu_back = abilities_from_race(p, V=V, D=D)
p_back = winning.race_probabilities(mu_back, V=V, D=D)
err = float(np.abs(np.log(p_back) - np.log(p)).max())
assert err < 1e-6, f"inversion round trip {err:.2e}"

# the Gumbel base IS softmax, which is the control variate claim. The
# identity is stated at unit temperature, where the Gumbel scale that
# matches softmax(-mu) is D = tau^2 pi^2 / 6, not D = 1.
p_gum = winning.race_probabilities(mu, D=np.full(n, np.pi ** 2 / 6.0),
                                   base="gumbel")
p_soft = softmax_probabilities(mu)
rel = float(np.abs(p_gum / p_soft - 1.0).max())
assert rel < 1e-9, f"gumbel vs softmax {rel:.2e}"

# dense covariance front door
C = 0.35 * np.ones((n, n)) + 0.65 * np.eye(n)
p_cov = winning.race_probabilities(mu, cov=C)
assert abs(float(p_cov.sum()) - 1.0) < 1e-10

# the classic API at its new home, and the deprecated alias still working
from winning.classic.lattice import skew_normal_density  # noqa: F401

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import winning.lattice  # noqa: F401
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "the legacy winning.lattice alias should warn"
    )

# ratings layer loads and runs one update
from winning.ratings.history import rate_history

ratings, logZ = rate_history(
    [{"t": 0.0, "runners": ["a", "b", "c"], "order": [1, 0, 2]}]
)
assert set(ratings) == {"a", "b", "c"}
assert np.isfinite(logZ)

print(
    f"smoke OK: {winning.__version__} | contract green | round trip {err:.1e} "
    f"| gumbel-softmax {rel:.1e}"
)
