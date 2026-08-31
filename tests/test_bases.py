"""The out-of-the-box base families: standardized, self-consistent, and
each verified against Monte Carlo in an actual race."""
import numpy as np
import pytest

from winning.factor.races import (BASES, failure_base, race_probabilities,
                                  skew_normal_base, student_base)

ALL = [("normal", "normal"), ("gumbel", "gumbel"),
       ("logistic", "logistic"), ("laplace", "laplace"),
       ("student4", student_base(4)), ("skew3", skew_normal_base(3)),
       ("skew-5", skew_normal_base(-5))]


@pytest.mark.parametrize("name,base", ALL)
def test_base_is_standardized_and_consistent(name, base):
    fn = BASES[base] if isinstance(base, str) else base
    x = np.linspace(-40, 40, 400001)
    dx = x[1] - x[0]
    S, f, fp = fn(x)
    assert abs(f.sum() * dx - 1) < 1e-5
    assert abs((x * f).sum() * dx) < 1e-6
    assert abs(((x ** 2) * f).sum() * dx - 1) < 2e-3
    # S' = -f and fp = f' (away from any kink)
    interior = np.abs(x) < 8
    assert np.abs(np.gradient(S, dx) + f)[interior].max() < 1e-4
    assert np.abs(np.gradient(f, dx) - fp)[interior].max() < 1e-4


@pytest.mark.parametrize("name,base", ALL)
def test_base_races_match_monte_carlo(name, base):
    rng = np.random.default_rng(3)
    n, M = 6, 400_000
    mu = np.sort(rng.normal(size=n))
    fn = BASES[base] if isinstance(base, str) else base
    xg = np.linspace(-60, 60, 800001)
    Sg, _, _ = fn(xg)
    u = rng.random((M, n))
    X = mu + np.interp(u, 1.0 - Sg, xg)
    counts = np.bincount(X.argmin(1), minlength=n) / M
    p = race_probabilities(mu, D=np.ones(n), base=base)
    assert 0.5 * np.abs(p - counts).sum() < 4e-3   # MC noise at 4e5 draws
    assert abs(p.sum() - 1) < 1e-9


def test_failure_base_races_match_monte_carlo():
    rng = np.random.default_rng(4)
    n, M, q = 6, 400_000, 0.2
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, D=np.ones(n), base=failure_base(q))
    perf = mu + rng.standard_normal((M, n))
    broke = rng.random((M, n)) < q
    perf[broke] += 6.0 + 0.35 * rng.standard_normal(int(broke.sum()))
    counts = np.bincount(perf.argmin(1), minlength=n) / M
    assert 0.5 * np.abs(p - counts).sum() < 4e-3


def test_order_pass_accepts_any_base():
    # bandits request: base= reached only the winner path, but a
    # retirement is a LAST-PLACE finish, so the failure lump matters
    # most with ranked feedback. Default must stay bit-identical, the
    # general path must match Monte Carlo, and the lump must split a
    # last place between "slow" and "broke".
    from winning.ratings.nway import _order_pass, update_order_correlated
    rng = np.random.default_rng(0)
    n = 5
    m = rng.normal(size=n)
    sd = 0.8 + 0.4 * rng.random(n)
    order = np.array(list(rng.permutation(n)))
    a = _order_pass(m, sd, order)
    b = _order_pass(m, sd, order, base="normal")
    assert a[0] == b[0] and np.array_equal(a[1], b[1])

    q = 0.2
    fb = failure_base(q)
    lp, gr = _order_pass(m, sd, order, base=fb, L=4001)
    M = 400_000
    Z = rng.standard_normal((M, n))
    broke = rng.random((M, n)) < q
    Z[broke] = 6.0 + 0.35 * rng.standard_normal(int(broke.sum()))
    X = m - sd * Z                       # max-wins: failure lands last
    hit = (np.argsort(-X, axis=1) == order).all(axis=1).mean()
    assert abs(np.exp(lp) - hit) < 4 * np.sqrt(max(hit, 1e-9) / M) + 1e-4
    eps = 1e-5
    mp = m.copy(); mp[3] += eps
    mm = m.copy(); mm[3] -= eps
    fd = (_order_pass(mp, sd, order, base=fb, L=4001)[0]
          - _order_pass(mm, sd, order, base=fb, L=4001)[0]) / (2 * eps)
    assert abs(fd - gr[3]) < 1e-4

    # the update: last place is penalized LESS when failures are modeled
    mm0 = np.zeros(n)
    o = np.arange(n)
    V0 = np.zeros((n, 1))
    g_mean, _, _ = update_order_correlated(mm0, np.ones(n), o, V0)
    f_mean, _, _ = update_order_correlated(mm0, np.ones(n), o, V0,
                                           base=failure_base(0.2))
    assert f_mean[-1] > g_mean[-1] + 0.1
    assert f_mean[0] < g_mean[0]


def test_failure_base_reaches_every_public_order_entry():
    """The order path is where a failure lump carries information, so
    every public order-side entry has to accept base= and act on it.

    Also locks the lattice-first property under a non-Gaussian base: with
    a DIAGONAL belief the full-covariance order update costs no quadrature
    and must reproduce the diagonal one exactly."""
    from winning.ratings.nway import (update_ranking, update_ranking_exact,
                                      order_loglik)
    from winning.ratings.full import update_order_full, update_winner_full
    from winning.ratings.market import update_race
    from winning.ratings.history import rate_history, predict_race

    fb = failure_base(0.2)
    n = 4
    m = np.array([0.3, 0.0, -0.2, 0.1])
    v = np.full(n, 0.4)
    order = [1, 0, 3, 2]
    sd = np.sqrt(v + 1.0)

    # every entry moves when the base changes (no silently ignored kwarg)
    assert not np.allclose(update_ranking(m, v, order, base=fb)[0],
                           update_ranking(m, v, order)[0])
    assert not np.allclose(update_ranking_exact(m, v, order, base=fb)[0],
                           update_ranking_exact(m, v, order)[0])
    assert not np.isclose(order_loglik(m, sd, order, base=fb)[0],
                          order_loglik(m, sd, order)[0])
    S = np.diag(v)
    assert not np.allclose(update_order_full(m, S, order, base=fb)[0],
                           update_order_full(m, S, order)[0])
    assert not np.allclose(update_race(m, v, order=order, base=fb)[0],
                           update_race(m, v, order=order)[0])

    # lattice-first: diagonal belief, non-Gaussian base, no quadrature
    # cost. The two paths agree to grid resolution rather than exactly --
    # they run the same recursion on different grids (_order_pass at
    # L=2001 against the batched pass, and the variance leg is a finite
    # difference either way), which under the Gaussian base costs nothing
    # in the mean and 2e-5 in the variance, and under the lump's sharper
    # features 2e-6 and half a percent.
    mf, Sf, _ = update_order_full(m, S, order, base=fb)
    md, vd = update_ranking_exact(m, v, order, base=fb)
    assert np.allclose(mf, md, atol=1e-5)
    assert np.allclose(np.diag(Sf), vd, rtol=1e-2)

    # history threads it, and the Gaussian-only winner path says so
    races = [{"t": 0.0, "runners": ["a", "b", "c", "d"], "order": order},
             {"t": 1.0, "runners": ["a", "b", "c", "d"], "order": [0, 2, 1, 3]}]
    rg, _ = rate_history(races)
    rf, _ = rate_history(races, base=fb)
    assert not np.isclose(rf["c"][0], rg["c"][0])
    _, _, st = rate_history(races, base=fb, return_state=True)
    p, _ = predict_race(st, ["a", "b", "c", "d"], base=fb)
    assert abs(float(np.sum(p)) - 1.0) < 1e-6
    with pytest.raises(NotImplementedError):
        update_winner_full(m, S, 0, base=fb)
    with pytest.raises(NotImplementedError):
        rate_history([{"t": 0.0, "runners": ["a", "b"], "winner": 0}],
                     base=fb)


def test_partial_order_is_exact_censoring():
    """Dropping a retirement from the finishing order marginalizes it
    out exactly: the partial order's likelihood is the sum over every
    position the omitted runner could have taken.

    This is the documented way to censor, and it is the recommended
    treatment when failures are independent of ability, so the identity
    is worth pinning rather than assuming."""
    from winning.ratings.nway import order_loglik, update_ranking_exact
    from winning.ratings.full import update_order_full

    m = np.array([0.4, 0.1, 0.0, -0.2, 0.3])
    v = np.full(5, 0.4)
    sd = np.sqrt(v + 1.0)
    part = [1, 0, 4, 3]                    # runner 2 retired

    lp = order_loglik(m, sd, part)[0]
    tot = sum(np.exp(order_loglik(m, sd, part[:k] + [2] + part[k:])[0])
              for k in range(5))
    assert abs(lp - np.log(tot)) < 1e-4

    # the censored runner's own belief is untouched under a diagonal prior
    md, vd = update_ranking_exact(m, v, part)
    assert md[2] == m[2] and vd[2] == v[2]
    assert not np.allclose(md[[0, 1, 3, 4]], m[[0, 1, 3, 4]])

    # under a correlated prior it moves, but only through real coupling:
    # ranking gradients sum to zero, so an equicorrelated prior carries
    # no information about a runner outside the order
    S_eq = np.full((5, 5), 0.15) + np.eye(5) * 0.25
    assert abs(update_order_full(m, S_eq, part)[0][2] - m[2]) < 1e-8
    S_one = np.eye(5) * 0.4
    S_one[1, 2] = S_one[2, 1] = 0.3
    assert abs(update_order_full(m, S_one, part)[0][2] - m[2]) > 0.1


def test_sharpness_dispatch_is_gauge_invariant_and_safe():
    """Two counterexamples from the eighth and ninth reviews, pinned.

    The raw row-norm statistic max |V_i|/sqrt(D_i) does NOT bound the
    pairwise contrast sharpness (centering can increase the largest row
    norm), so a dispatcher trusting it can stay with Gauss-Hermite on a
    genuinely sharp race. The engine gauge-fixes V <- PV and dispatches
    on sqrt(2) max |(PV)_i|/sqrt(D_i) >= s_pairwise, which cannot miss.
    """
    import warnings
    from winning.factor.races import race_probabilities

    # ninth review: three runners, analytic zero-mean orthant referee
    V = np.array([[-2.9, 0.0], [2.9, 0.01], [2.9, -0.01]])
    D = np.ones(3)
    mu = np.zeros(3)
    S = V @ V.T + np.diag(D)
    p_exact = []
    for i in range(3):
        d = [np.eye(3)[i] - np.eye(3)[j] for j in range(3) if j != i]
        var = [dd @ S @ dd for dd in d]
        rho = (d[0] @ S @ d[1]) / np.sqrt(var[0] * var[1])
        p_exact.append(0.25 + np.arcsin(rho) / (2 * np.pi))
    p_exact = np.array(p_exact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = race_probabilities(mu, V=V, D=D)
    assert 0.5 * np.abs(p - p_exact).sum() < 5e-5, p

    # eighth review: four runners; raw row norm 2.977 hides centered 4.43
    V4 = np.array([[2.95, 0.0], [2.95, 0.0], [2.95, 0.4], [-2.95, 0.0]])
    mu4 = np.zeros(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p4 = race_probabilities(mu4, V=V4, D=np.ones(4))
        # gauge invariance: a common loading column is choice-irrelevant
        # and after the internal centering the answer is bit-identical
        c = np.array([5.0, -3.0])
        p4s = race_probabilities(mu4, V=V4 + np.ones((4, 1)) @ c[None, :],
                                 D=np.ones(4))
    assert np.array_equal(p4, p4s)
    pmc = np.array([0.1831662, 0.1832104, 0.1908283, 0.4427951])  # 20M draws
    assert 0.5 * np.abs(p4 - pmc).sum() < 3e-4
