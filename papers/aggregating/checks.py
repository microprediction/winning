"""Numerical checks for the note on the Aggregating Algorithm for races.

E1  The substitution s(P) = inverse_race(E_P[race(x)]) meets the 1-mixability
    inequality of the winner log-loss with equality, for non-Gaussian P.
E2  For Gaussian P = N(m, S), E_P[race(x)] is the race at m with the
    covariance I + S, one forward call (S low rank plus diagonal).
E3  Online: a Gaussian (Laplace) approximation of the aggregating posterior,
    predicting by the mixed race, against the plug-in race at the posterior
    mean and the best fixed abilities in hindsight.

Conventions: lower ability wins; D is the idiosyncratic variance; V holds
factor loadings, so the performance covariance is V V' + diag(D).
Seeded; writes results.json next to this file.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from winning.factor.races import abilities_from_race, race_probabilities

HERE = Path(__file__).resolve().parent
OUT = {}


def e1(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (3, 5, 10, 25):
        for K in (2, 7, 50):
            X = rng.standard_normal((K, n)) * rng.uniform(0.3, 2.0)
            X[: K // 2] += rng.standard_normal(n)        # lumpy, non-Gaussian
            w = rng.dirichlet(np.ones(K))
            pbar = sum(wk * race_probabilities(x) for wk, x in zip(w, X))
            s = abilities_from_race(pbar)
            ps = race_probabilities(s)
            # mixability with alpha = 1: -ln ps_y  <=  -ln E_P[p_y(x)] = -ln pbar_y
            gap = float(np.max(np.abs(np.log(ps) - np.log(pbar))))
            rows.append(dict(n=n, K=K, max_abs_loss_gap=gap, min_pbar=float(pbar.min())))
    OUT["E1"] = rows
    print("E1 max |loss(s(P),y) + ln E_P p_y| over all y:",
          max(r["max_abs_loss_gap"] for r in rows))


def e2(seed=2, N=4_000_000):
    rng = np.random.default_rng(seed)
    rows = []
    for n, r in ((4, 1), (8, 2), (12, 2)):
        m = rng.standard_normal(n) * 0.7
        L = rng.standard_normal((n, r)) * 0.6            # posterior factor part
        d = rng.uniform(0.05, 0.8, n)                    # posterior diagonal part
        # mixture of unit-variance races over x ~ N(m, L L' + diag d)
        one = race_probabilities(m, V=L, D=1.0 + d)
        t = time.perf_counter()
        one = race_probabilities(m, V=L, D=1.0 + d)
        sec = time.perf_counter() - t
        X = m + rng.standard_normal((N, r)) @ L.T + np.sqrt(d) * rng.standard_normal((N, n))
        perf = X + rng.standard_normal((N, n))
        mc = np.bincount(perf.argmin(1), minlength=n) / N
        se = np.sqrt(mc * (1 - mc) / N)
        rows.append(dict(n=n, rank=r, max_abs_diff=float(np.abs(one - mc).max()),
                         max_z=float(np.max(np.abs(one - mc) / se)), seconds=sec))
    OUT["E2"] = rows
    for row in rows:
        print("E2", row)


def _loglik(x, y):
    return float(np.log(race_probabilities(x)[y]))


def _grad_hess_diag(x, y, h=1e-3):
    f0 = _loglik(x, y)
    g = np.empty(len(x)); H = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x)); e[i] = h
        fp, fm = _loglik(x + e, y), _loglik(x - e, y)
        g[i] = (fp - fm) / (2 * h); H[i] = (fp - 2 * f0 + fm) / h ** 2
    return g, H


def e3(seed=3, n=20, k=5, T=1000, prior_var=1.0):
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal(n)
    fields, winners = [], []
    for _ in range(T):
        F = rng.choice(n, k, replace=False)
        p = race_probabilities(truth[F])
        fields.append(F); winners.append(int(rng.choice(k, p=p)))
    # Gaussian aggregating posterior, diagonal, one Laplace step per race
    m = np.zeros(n); s = np.full(n, prior_var)
    loss_mix = loss_plug = 0.0
    curve = []
    t0 = time.perf_counter()
    for t, (F, y) in enumerate(zip(fields, winners)):
        p_mix = race_probabilities(m[F], D=1.0 + s[F])     # E over N(m,S): one call
        p_plug = race_probabilities(m[F])                   # posterior mean only
        loss_mix -= np.log(p_mix[y]); loss_plug -= np.log(p_plug[y])
        g, H = _grad_hess_diag(m[F], y)
        prec = 1.0 / s[F] - np.minimum(H, 0.0)
        s_new = 1.0 / prec
        m[F] = m[F] + s_new * g
        s[F] = s_new
        if (t + 1) % 100 == 0:
            curve.append(dict(t=t + 1, mix=float(loss_mix), plug=float(loss_plug)))
    sec = time.perf_counter() - t0
    # best fixed abilities in hindsight: maximum likelihood, the gradient
    # summed race by race over each field's own coordinates
    def nll_grad(x):
        f = 0.0; G = np.zeros(n)
        for F, y in zip(fields, winners):
            f -= _loglik(x[F], y)
            g, _ = _grad_hess_diag(x[F], y)
            G[F] -= g
        return f, G
    def nll(x):
        return -sum(_loglik(x[F], y) for F, y in zip(fields, winners))
    best = minimize(nll_grad, m.copy(), jac=True, method="L-BFGS-B",
                    options=dict(maxiter=300, gtol=1e-6))
    loss_best = float(best.fun)
    loss_truth = float(nll(truth))
    OUT["E3"] = dict(n=n, k=k, T=T, loss_mixed=float(loss_mix), loss_plugin=float(loss_plug),
                     loss_best_fixed=loss_best, loss_truth=loss_truth,
                     regret_mixed=float(loss_mix - loss_best),
                     regret_plugin=float(loss_plug - loss_best),
                     seconds_online=sec, curve=curve)
    print("E3", {k_: v for k_, v in OUT["E3"].items() if k_ != "curve"})


def _grad_hess_full(x, y, h=1e-3):
    k = len(x); f0 = _loglik(x, y)
    E = np.eye(k) * h
    fp = np.array([_loglik(x + E[i], y) for i in range(k)])
    fm = np.array([_loglik(x - E[i], y) for i in range(k)])
    g = (fp - fm) / (2 * h)
    H = np.empty((k, k))
    for i in range(k):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, k):
            fpp = _loglik(x + E[i] + E[j], y); fmm = _loglik(x - E[i] - E[j], y)
            fpm = _loglik(x + E[i] - E[j], y); fmp = _loglik(x - E[i] + E[j], y)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    return g, H


def e5(seeds=(3, 13, 23, 33, 43), n=20, k=5, T=1000, prior_var=1.0, key="E5"):
    """Online, several seeds: full-covariance Gaussian aggregating posterior
    (one Laplace step per race on the field's block) predicting by the
    mixed race p_{I + S_FF}(m_F), against the diagonal posterior and the
    plug-in race at the posterior mean. Losses in nats, summed over T."""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth = rng.standard_normal(n)
        m = np.zeros(n); S = np.eye(n) * prior_var
        md = np.zeros(n); sd = np.full(n, prior_var)
        L = dict(full_mixed=0.0, full_plugin=0.0, diag_mixed=0.0, diag_plugin=0.0, truth=0.0)
        for _ in range(T):
            F = rng.choice(n, k, replace=False)
            p = race_probabilities(truth[F]); y = int(rng.choice(k, p=p))
            L["truth"] -= np.log(p[y])
            SF = S[np.ix_(F, F)]
            C = np.linalg.cholesky(SF + 1e-12 * np.eye(k))
            L["full_mixed"] -= np.log(race_probabilities(m[F], V=C, D=np.ones(k))[y])
            L["full_plugin"] -= np.log(race_probabilities(m[F])[y])
            L["diag_mixed"] -= np.log(race_probabilities(md[F], D=1.0 + sd[F])[y])
            L["diag_plugin"] -= np.log(race_probabilities(md[F])[y])
            # full Laplace step: precision gains -H on the field block
            g, H = _grad_hess_full(m[F], y)
            H = 0.5 * (H + H.T)
            w, U = np.linalg.eigh(-H); A = (U * np.maximum(w, 0.0)) @ U.T   # PSD part
            # rank-k update of S through the field block (Woodbury)
            SFcols = S[:, F]
            K = np.linalg.solve(np.eye(k) + A @ SF, A)                       # (I + A S_FF)^{-1} A
            S = S - SFcols @ K @ SFcols.T
            m = m + S[:, F] @ g
            # diagonal Laplace, as in E3
            gd, Hd = _grad_hess_diag(md[F], y)
            sd_new = 1.0 / (1.0 / sd[F] - np.minimum(Hd, 0.0))
            md[F] = md[F] + sd_new * gd; sd[F] = sd_new
        rows.append(dict(seed=seed, **{k_: float(v) for k_, v in L.items()}))
        print(key, rows[-1])
    diffs = {key: [r[key] - r["full_plugin"] for r in rows]
             for key in ("full_mixed", "diag_mixed", "diag_plugin")}
    OUT[key] = dict(n=n, k=k, T=T, rows=rows,
                     mean_minus_full_plugin={key: float(np.mean(v)) for key, v in diffs.items()},
                     se={key: float(np.std(v, ddof=1) / np.sqrt(len(v))) for key, v in diffs.items()})
    print(key, "mean vs full plug-in", OUT[key]["mean_minus_full_plugin"], OUT[key]["se"])


def e6():
    """Cold start: 400 entrants, about 12 races each."""
    e5(n=400, key="E6")


def e4(seed=4, m_samples=100, reps=5):
    """Large fields: the mixed race in one call, against averaging m_samples
    plug-in races over posterior draws (the Monte Carlo step of GAF,
    Jezequel et al. 2021, which used m = 100)."""
    rng = np.random.default_rng(seed)
    rows = []
    for n, r in ((100, 2), (1000, 2), (5000, 3)):
        m = rng.standard_normal(n) * 0.8
        L = rng.standard_normal((n, r)) * 0.4
        d = rng.uniform(0.05, 0.5, n)
        t = time.perf_counter()
        exact = race_probabilities(m, V=L, D=1.0 + d)
        sec_exact = time.perf_counter() - t
        errs, secs = [], []
        for _ in range(reps):
            t = time.perf_counter()
            X = m + rng.standard_normal((m_samples, r)) @ L.T + np.sqrt(d) * rng.standard_normal((m_samples, n))
            mc = np.mean([race_probabilities(x) for x in X], axis=0)
            secs.append(time.perf_counter() - t)
            # relative error on the favourites, where the mass is
            top = np.argsort(-exact)[:10]
            errs.append(float(np.max(np.abs(mc[top] - exact[top]) / exact[top])))
        rows.append(dict(n=n, rank=r, seconds_exact=sec_exact,
                         seconds_mc=float(np.median(secs)),
                         mc_max_rel_err_top10=float(np.median(errs)),
                         top_prob=float(exact.max())))
        print("E4", rows[-1])
    OUT["E4"] = rows


if __name__ == "__main__":
    import sys
    path = HERE / "results.json"
    if path.exists():
        OUT.update(json.loads(path.read_text()))
    for name in (sys.argv[1:] or ["e1", "e2", "e3", "e4"]):
        globals()[name]()
    path.write_text(json.dumps(OUT, indent=1))
