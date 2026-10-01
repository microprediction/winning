"""The GHK route for a degraded cov= fit is permutation-equivariant,
accurate to its regression tolerance, and inverted by the same map
(#162, #164, #158).

GHK conditions the runners sequentially, so its error depends on the
order they are listed in: one 8-runner field priced 8.8e-3 apart under
two labelings, and 5.6e-3 from a 12M-path Monte Carlo reference at 1024
nodes. The route now sorts the runners canonically before the estimate
(equivariance exact) and spends 4096 nodes (7e-4 on that field). The
inverse under cov= used to iterate against the fitted lattice while the
forward returned GHK, so the two front doors described different races,
4e-3 to 8e-3 apart on exact rank-1/3/5 fixtures; it now inverts the
routed map with own-slopes GHK accumulates in its conditioning pass."""
import warnings

import numpy as np
import pytest

import winning.factor as wf
import winning.methods
from winning.factor.races import _fit_cov


def _issue_162_field():
    rng = np.random.default_rng(314159)
    for _ in range(217):
        n = 8
        A = rng.normal(size=(n, n))
        ridge = 10 ** rng.uniform(-4, 0.5)
        S = A @ A.T + ridge * n * np.eye(n)
        d = np.sqrt(np.diag(S))
        C = S / np.outer(d, d)
        mu = rng.normal(scale=10 ** rng.uniform(-0.3, 1), size=n)
    return mu, C


# 12,000,000-path direct Monte Carlo on the field above (#162; largest
# standard error 1.4e-4)
_MC_162 = np.array([6.67e-07, 2.5e-06, 0.3048739, 0.0015809, 0.0554644,
                    0.0, 0.0, 0.6380776])


def _exact_correlation(n, r, seed):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, r))
    d = 0.05 + rng.random(n)
    S = V @ V.T + np.diag(d)
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def test_the_162_field_is_routed():
    mu, C = _issue_162_field()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, _, degraded = _fit_cov(C, None, None, None, will_route=True)
    assert degraded


@pytest.mark.parametrize("perm", [
    np.array([1, 3, 4, 6, 0, 5, 2, 7]),
    np.array([7, 6, 5, 4, 3, 2, 1, 0]),
    np.array([2, 0, 1, 3, 4, 5, 6, 7]),
])
def test_relabeling_the_runners_relabels_the_answer(perm):
    mu, C = _issue_162_field()
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # the route is silent
        p = np.asarray(wf.race_probabilities(mu, cov=C))
        pp = np.asarray(wf.race_probabilities(mu[perm], cov=C[np.ix_(perm, perm)]))
    back = np.empty(8)
    back[perm] = pp
    assert np.abs(back - p).max() < 1e-12


def test_the_route_meets_its_tolerance_against_monte_carlo():
    mu, C = _issue_162_field()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(wf.race_probabilities(mu, cov=C))
    assert np.abs(p - _MC_162).max() < 2e-3        # 7.2e-4 measured; 5.6e-3 before


@pytest.mark.parametrize("r,seed", [(1, 1), (3, 3), (5, 5)])
def test_forward_and_inverse_describe_the_same_race(r, seed):
    """#164's fixtures: the public inverse of the public forward must
    round-trip, and recover the abilities that produced the target."""
    N = 8
    mu0 = np.linspace(-0.6, 0.6, N)
    C = _exact_correlation(N, r, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = wf.race_probabilities(mu0, cov=C)
        mu, info = wf.abilities_from_race(p, cov=C, return_info=True)
        q = wf.race_probabilities(mu, cov=C)
    assert info["converged"], info
    assert np.abs(q - p).max() < 1e-8
    assert np.abs(mu - (mu0 - mu0.mean())).max() < 1e-7


def test_the_162_field_round_trips():
    mu0, C = _issue_162_field()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = wf.race_probabilities(mu0, cov=C)
        mu, info = wf.abilities_from_race(p, cov=C, return_info=True)
        q = wf.race_probabilities(mu, cov=C)
    assert info["converged"], info
    assert np.abs(q - p).max() < 1e-8


def test_ghk_own_slopes_track_central_differences():
    """The preconditioner the inverse uses: GHK's in-pass own-slope drops
    the truncated draws' dependence on mu, so it is biased a little
    large, and must stay within the factor the adaptive damping absorbs."""
    from winning.factor.races import _race_dense
    rng = np.random.default_rng(3)
    n = 6
    A = rng.normal(size=(n, n))
    C = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(C))
    C /= np.outer(d, d)
    m = rng.normal(size=n)
    _, sl = _race_dense(m, C, return_slopes=True)
    eye = np.eye(n)
    fd = np.array([(_race_dense(m + 1e-4 * eye[i], C)[i]
                    - _race_dense(m - 1e-4 * eye[i], C)[i]) / 2e-4
                   for i in range(n)])
    assert np.all(sl < 0) and np.all(fd < 0)
    assert np.all(np.abs(sl / fd - 1.0) < 0.35)
    lp, dl = _race_dense(m, C, return_slopes=True, log=True)
    p = _race_dense(m, C)
    assert np.abs(np.exp(lp) - p).max() < 1e-12
    assert np.abs(dl * p - sl).max() < 1e-12


def test_ghk_log_probabilities_stay_finite_where_p_underflows():
    """A runner far behind on a sharp field: p underflows to 0, log p is a
    real number the Newton step can use (#164's n=30 warm start)."""
    from winning.factor.races import _race_dense
    mu = np.array([0.0, 0.0, 60.0])
    C = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = _race_dense(mu, C)
    lp = _race_dense(mu, C, log=True)
    assert p[2] == 0.0
    assert np.isfinite(lp[2]) and lp[2] < -700


def test_the_degeneration_warning_names_a_recipe_that_runs():
    """#158: the warning used to recommend winning.methods.qmc_ghk (not an
    attribute) with V=chol(cov) and no D (TypeError). Whatever it names
    now must execute as written."""
    import re
    C = _exact_correlation(8, 3, 3)
    mu = np.linspace(-0.5, 0.5, 8)
    with pytest.warns(RuntimeWarning) as rec:
        wf.race_probabilities(mu, cov=C, return_slopes=True)
    msgs = [str(w.message) for w in rec if "degenerated" in str(w.message)]
    assert msgs, [str(w.message) for w in rec]
    msg = msgs[0]
    assert "route" in msg and "automatically" in msg
    call = re.search(r"winning\.methods\.get_method\('qmc_ghk'\)\((.*?)\)\s\(max-wins\)", msg)
    assert call, msg
    n = len(mu)
    numpy = np
    p, info = eval(f"winning.methods.get_method('qmc_ghk')({call.group(1)})",
                   {"winning": winning, "numpy": numpy, "mu": mu, "cov": C, "n": n})
    assert abs(p.sum() - 1) < 1e-12 and "logp" in info


import winning  # noqa: E402  (used by the eval above)
