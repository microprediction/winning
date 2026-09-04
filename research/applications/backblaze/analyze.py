"""First-failure correlation on Backblaze: is the independent MTBF
model falsified, and by how much does it mis-state k-out-of-n risk.

From the compact (date, model) cohort table:
  1. Validate extraction: per-model annualized failure rate against
     Backblaze's published order of magnitude (~1-2%).
  2. Overdispersion test -- the independence falsification. Under
     independent constant hazard, a cohort's daily failure count is
     Poisson with variance equal to its mean. Common-cause events
     (bad batch aging in, firmware, heat) cluster failures and inflate
     the variance. The dispersion index Var/Mean > 1 falsifies
     independence; = 1 supports it.
  3. One-factor common-cause fit -- the engine's structure. Model the
     daily failure count of a cohort as binomial with a shared latent
     daily log-hazard: conditional on z_t ~ N(0,1),
     p_t = Phi(mu + sigma z_t), failures_t ~ Bin(cohort_t, p_t).
     Fit (mu, sigma) by marginal ML (Gauss-Hermite over z_t). sigma
     is the common-cause correlation strength; sigma = 0 is
     independence.
  4. Consequence for k-out-of-n durability. For a cohort of n drives
     over the window, P(at least k failures) under the fitted
     correlated model vs the independent one. Common cause makes
     clustered failures likelier, so the LARGE-k (data-loss) tail is
     HEAVIER than independence predicts: the independent MTBF model
     UNDER-estimates correlated data-loss risk (the mirror of the
     latency case, where positive correlation makes the max tail
     lighter and independence over-provisions).
"""
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = json.load(open(os.path.join(HERE, "cohort_table.json")))
    rows = {}
    for k, (n, f) in d.items():
        day, model = k.split("|", 1)
        rows.setdefault(model, {})[day] = (n, f)
    return rows


def factor_fit(cohorts, fails):
    """Fit (mu, sigma) of failures_t ~ Bin(cohort_t, Phi(mu+sigma z))."""
    zq, wq = np.polynomial.hermite_e.hermegauss(41)
    wq = wq / wq.sum()
    c = np.asarray(cohorts, float)
    f = np.asarray(fails, float)
    logC = gammaln(c + 1) - gammaln(f + 1) - gammaln(c - f + 1)

    def nll(par):
        mu, ls = par
        p = norm.cdf(mu + np.exp(ls) * zq)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = (logC[:, None] + f[:, None] * np.log(p)[None, :]
              + (c - f)[:, None] * np.log1p(-p)[None, :])
        return -(logsumexp(ll + np.log(wq)[None, :], axis=1)).sum()

    best = None
    for mu0 in (-3.5, -3.0, -2.5):
        r = minimize(nll, np.array([mu0, -1.0]), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    return best.x[0], float(np.exp(best.x[1])), zq, wq


def kout_tail(n, mu, sigma, zq, wq, ks):
    """P(>= k failures among n) with shared factor: mixture of
    Binomials over z, exact via the count pgf per node."""
    out = {}
    # per node, Binomial(n, p_z) pmf via FFT-free cumulative
    from scipy.stats import binom
    for k in ks:
        acc = 0.0
        for z, w in zip(zq, wq):
            p = np.clip(norm.cdf(mu + sigma * z), 1e-12, 1 - 1e-12)
            acc += w * binom.sf(k - 1, n, p)
        out[k] = float(acc)
    return out


if __name__ == "__main__":
    rows = load()
    # rank models by drive-days
    ranked = sorted(rows.items(),
                    key=lambda kv: -sum(n for n, _ in kv[1].values()))
    print("top cohorts by drive-days:")
    disp = []
    fleet_days = {}
    for model, days in ranked[:12]:
        dd = sum(n for n, _ in days.values())
        ff = sum(f for _, f in days.values())
        afr = 365.0 * ff / dd if dd else 0.0
        ordered = sorted(days.items())
        daily_f = np.array([f for _, (n, f) in ordered], float)
        cohort = np.array([n for _, (n, f) in ordered], float)
        di = (daily_f.var() / daily_f.mean()
              if daily_f.mean() > 0 else float("nan"))
        disp.append(di)
        # per-model one-factor sigma (common-cause within this batch)
        try:
            _, sig_m, _, _ = factor_fit(cohort, daily_f)
        except Exception:
            sig_m = float("nan")
        print(f"  {model:22s} drive-days {dd:9d} fails {ff:4d} "
              f"AFR {afr*100:4.1f}%  Var/Mean {di:5.2f}  sigma "
              f"{sig_m:.2f}")
    print(f"median dispersion index over top cohorts: "
          f"{np.nanmedian(disp):.2f} (1.0 = independent Poisson)")

    # fleet daily counts and factor fit
    all_days = {}
    for model, days in rows.items():
        for day, (n, f) in days.items():
            a, b = all_days.get(day, (0, 0))
            all_days[day] = (a + n, b + f)
    dd_sorted = sorted(all_days.items())
    cohorts = [v[0] for _, v in dd_sorted]
    fails = [v[1] for _, v in dd_sorted]
    fleet_di = np.var(fails) / np.mean(fails)
    mu, sigma, zq, wq = factor_fit(cohorts, fails)
    print(f"\nfleet daily failures: dispersion {fleet_di:.1f}; "
          f"one-factor fit mu={mu:.3f} sigma={sigma:.3f} "
          f"(sigma=0 is independence)")

    # k-out-of-n consequence on a representative durability set
    n_set = 20
    base_p = norm.cdf(mu)
    ks = [1, 2, 3, 4, 6]
    corr = kout_tail(n_set, mu, sigma, zq, wq, ks)
    indep = kout_tail(n_set, mu, 0.0, zq, wq, ks)  # sigma=0 -> point p
    print(f"\nk-out-of-{n_set} failures in one day (per-drive p="
          f"{base_p:.4f}):")
    for k in ks:
        ratio = corr[k] / indep[k] if indep[k] > 0 else float("inf")
        print(f"  P(>= {k}): correlated {corr[k]:.2e}  independent "
              f"{indep[k]:.2e}  ratio {ratio:.1f}x")

    json.dump(dict(median_dispersion=float(np.nanmedian(disp)),
                   fleet_dispersion=float(fleet_di),
                   factor=dict(mu=mu, sigma=sigma),
                   kout=dict(correlated=corr, independent=indep,
                             n=n_set)),
              open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("wrote results.json")
