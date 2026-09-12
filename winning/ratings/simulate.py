"""Generative models for the ratings layer: the one place synthetic
abilities, contests and base noise are drawn, shared by the tests, the
verifier (winning.ratings.verify) and research scripts.

Conventions. The engine's bases are standardized MIN-wins laws: a base
callable returns (S, f, fp) with S(z) = P(Z > z). The ratings modules
are MAX-wins, with performance x = s + sqrt(beta2) * eps and eps the
base variable flipped, density f_min(-z). `sample_noise` therefore
returns MAX-wins draws (eps = -Z), and every callable base carries a
`.sample(rng, size)` attribute drawing Z in its OWN min-wins convention
(attached by the factories in winning.factor.races). The density that
defines a base defines its sampler, so a referee built on these draws
cannot drift from the engine; `check_sampler` pins each one to its
survival function by a Kolmogorov distance.

Worlds reproduce, draw for draw, the private helpers they replace
(tests/test_correlated_updates.py, tests/test_history_and_teams.py,
tests/test_factor_ratings.py) and the bandits audit's `_world/_events`
(independent_world with noise="gaussian"), so pinned numbers do not move.
"""

from __future__ import annotations

import numpy as np

_EULER = 0.5772156649015329
_C_GUMBEL = np.pi / np.sqrt(6.0)


# ---------------------------------------------------------------------------
# base noise
# ---------------------------------------------------------------------------

def sample_min(rng, base, size=None):
    """Draws of the standardized MIN-wins base variable Z (survival S(z)
    = P(Z > z)): the base's own convention. Named bases by closed form;
    callable bases through their `.sample` attribute."""
    if callable(base):
        fn = getattr(base, "sample", None)
        if fn is None:
            raise ValueError("callable base has no .sample(rng, size); use a "
                             "winning.factor.races factory or attach one")
        return np.asarray(fn(rng, size), dtype=float)
    if base == "normal":
        return rng.standard_normal(size)
    if base == "logistic":
        return rng.logistic(0.0, np.sqrt(3.0) / np.pi, size)
    if base == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    if base == "gumbel":
        # S(z) = exp(-exp(c z - gamma)) is Z = (gamma - X) / c with X a
        # standard max-Gumbel (winning.factor.races._gumbel_min)
        return (_EULER - rng.gumbel(0.0, 1.0, size)) / _C_GUMBEL
    raise ValueError(f"unknown base {base!r}")


def sample_noise(rng, base, size=None):
    """MAX-wins standardized noise eps = -Z: the ratings modules'
    f_min(-z) convention (research/adjudications/predictive_referee.py
    carried this for the named bases; callables now sample too)."""
    return -sample_min(rng, base, size)


def base_survival(base, z):
    """S(z) of a base, named or callable (min-wins)."""
    from ..factor.races import BASES
    fn = base if callable(base) else BASES[base]
    return np.asarray(fn(np.asarray(z, dtype=float))[0], dtype=float)


def check_sampler(base, rng, n=200_000):
    """Kolmogorov distance between the empirical CDF of `n` min-wins
    draws and 1 - S(z) from the base itself. Exact samplers sit at the
    DKW noise floor (~1.2/sqrt(n) at 99 percent); a sign or scale slip
    is orders of magnitude above it."""
    z = np.sort(sample_min(rng, base, n))
    cdf = 1.0 - base_survival(base, z)
    i = np.arange(1, n + 1) / n
    return float(max(np.abs(cdf - i).max(), np.abs(cdf - (i - 1.0 / n)).max()))


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------

def correlated_draws(rng, M, m, v, V=None, beta2=1.0, base="normal"):
    """M joint draws of (skill, performance) from the correlated model:
    s ~ N(m, v), f ~ N(0, I_r), x = s + V f + sqrt(beta2) eps, eps the
    max-wins base noise. Returns (s, x), each (M, n). With base="normal"
    and V given this is tests/test_correlated_updates.py::_simulate."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(m)
    s = m + np.sqrt(v) * rng.normal(size=(M, n))
    x = s.copy()
    if V is not None:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if V.shape[0] != n:
            V = V.T
        f = rng.normal(size=(M, V.shape[1]))
        x = x + f @ V.T
    b2 = np.broadcast_to(np.asarray(beta2, dtype=float), (n,))
    x = x + np.sqrt(b2) * sample_noise(rng, base, (M, n))
    return s, x


def independent_world(rng, M=8, K=5, n=40, prior_var=1.0, beta2=1.0, V=None,
                      base="normal", noise="base"):
    """The audit world: M entities with abilities a ~ N(0, prior_var),
    mean-centred, and n contests of K distinct entrants with performance
    a[S] + (V[S] f)_j + sqrt(beta2) eps, order best first. noise="base"
    draws eps from the base under test (the model's own generative
    assumption); noise="gaussian" is the bandits audit's world (Gaussian
    noise under every base), kept as a labelled robustness regime and
    reproducing its draws exactly at prior_var = beta2 = 1, V = None.
    Returns (a, events) with events a list of (S, order)."""
    a = rng.normal(0.0, 1.0, M) * np.sqrt(float(prior_var))
    a = a - a.mean()
    if V is not None:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if V.shape[0] != M:
            V = V.T
    sb = np.sqrt(float(beta2))
    events = []
    for _ in range(n):
        S = rng.choice(M, K, replace=False)
        perf = a[S].copy()
        if V is not None:
            perf = perf + V[S] @ rng.normal(size=V.shape[1])
        if noise == "gaussian":
            perf = perf + sb * rng.normal(0.0, 1.0, K)
        else:
            perf = perf + sb * sample_noise(rng, base, K)
        events.append((S, np.argsort(-perf)))
    return a, events


def history_world(rng, n_h, n_races, drift_ts, easing=None,
                  observe=("p_market", "margins"), field=6,
                  lengths_scale=0.2, market_sd=0.3):
    """OU-drifting abilities with market prices and finishing margins:
    tests/test_history_and_teams.py::_make_history, draw for draw.
    observe selects which keys each race carries ("p_market", "margins",
    "order", "winner"); the draws are identical whatever is emitted.
    Returns (races, truth_at_end) in rate_history's dict format, with
    margins in lengths at 1/lengths_scale per performance unit."""
    from ..factor.races import race_probabilities
    s = rng.normal(size=n_h)
    lam = np.exp(-1.0 / drift_ts)
    races, truth_at_end = [], None
    for t in range(n_races):
        s = lam * s + np.sqrt(1 - lam ** 2) * rng.normal(size=n_h)
        runners = list(rng.choice(n_h, size=field, replace=False))
        st = s[runners]
        y_mkt = st + market_sd * rng.normal(size=field)
        p_mkt = race_probabilities(-y_mkt)
        perf = st + rng.normal(size=field)
        gaps = perf.max() - perf
        L = gaps if easing is None else easing(gaps)
        race = dict(t=float(t), runners=runners)
        if "p_market" in observe:
            race["p_market"] = p_mkt
        if "margins" in observe:
            race["margins"] = L / lengths_scale
        if "order" in observe:
            race["order"] = list(np.argsort(-perf))
        if "winner" in observe:
            race["winner"] = int(np.argmax(perf))
        races.append(race)
        truth_at_end = s.copy()
    return races, truth_at_end


def factor_world(rng, n_ent, K, n_events, off_sd=0.4, n_cov=2):
    """Entities with a level plus (n_cov - 1) offsets, contests of K
    entrants under exogenously drawn binary conditions:
    tests/test_factor_ratings.py::_factor_world. Returns (B, events) in
    fit_factor_ratings' (subset, order, x) format."""
    B = np.column_stack([rng.normal(size=n_ent)]
                        + [rng.normal(size=n_ent) * off_sd
                           for _ in range(n_cov - 1)])
    evs = []
    for _ in range(n_events):
        sub = rng.choice(n_ent, K, replace=False)
        x = np.concatenate([[1.0], (rng.random(n_cov - 1) < 0.5).astype(float)])
        perf = B[sub] @ x + rng.normal(size=K)
        evs.append((sub, np.argsort(-perf), x))
    return B, evs


def block_world(seed, n_ent=12, n_contests=40, rho=0.5, K=6):
    """Two groups of K/2 sharing a performance component with share rho,
    variance preserving: tests/test_factor_ratings.py::_block_world.
    `seed` may be an int or a Generator. Returns tracker-format contests
    carrying 'groups'."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    half = K // 2
    s = rng.normal(size=n_ent)
    contests = []
    for k in range(n_contests):
        ids = rng.choice(n_ent, K, replace=False)
        z = rng.normal(size=2)
        gidx = np.array([0] * half + [1] * (K - half))
        x = (s[ids] + np.sqrt(rho) * z[gidx]
             + np.sqrt(1 - rho) * rng.normal(size=K))
        contests.append({"runners": [str(i) for i in ids], "t": float(k),
                         "order": list(np.argsort(-x)),
                         "groups": ["A"] * half + ["B"] * (K - half)})
    return contests


def mixed_events(rng, n_feat=7):
    """One design event of every shape the factor fitter handles: K=2
    full and winner-only, K=4 full, K=5 and K=4 partial, K=3 winner-only,
    and a sparse design (tests/test_factor_ratings.py::_mixed_events)."""
    from scipy import sparse
    evs = [(rng.normal(size=(2, n_feat)), [1, 0]),
           (rng.normal(size=(2, n_feat)), [0]),
           (rng.normal(size=(4, n_feat)), [2, 0, 3, 1]),
           (rng.normal(size=(5, n_feat)), [4, 1]),
           (rng.normal(size=(3, n_feat)), [1]),
           (rng.normal(size=(4, n_feat)), [1, 3])]
    Zs = rng.normal(size=(4, n_feat)) * (rng.random((4, n_feat)) < 0.5)
    evs.append((sparse.csr_matrix(Zs), [3, 2, 1, 0]))
    return evs
