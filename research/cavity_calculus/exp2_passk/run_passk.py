"""Pass@k prediction: plug-in independence vs latent-difficulty models.

TailSFT's diagnostic estimates pass@16 from pass@1 by
f_k(p) = 1 - (1-p)^k, the plug-in independence extrapolation. On the
released Pass8-Rollouts records (9k prompts x 8 scored samples,
Qwen3-4B-Thinking on RLVE environments) this experiment measures what
that formula costs against hierarchical latent-difficulty models, on
held-out samples.

Design: samples 0-3 train, 4-7 test, per prompt. The held-out
per-prompt pass@4 is a single Bernoulli (any success among the four),
so predicted probabilities are scored by log loss and Brier with no
estimator ambiguity. Predictors of P(any success in 4 | 4 training
samples):
  plugin   1 - (1 - s'/4)^4               (their formula's structure)
  probit   posterior predictive under s' ~ Bin(4, Phi(theta)),
           theta ~ N(m, tau^2), (m, tau) by marginal ML, quadrature
  betabin  posterior predictive under a Beta(a, b) difficulty prior,
           (a, b) by marginal ML
Aggregate extrapolation: predicted mean pass@k for k = 1..8 from the
training half, against the unbiased empirical estimator
1 - C(8-s, k)/C(8, k) averaged over prompts (all 8 samples; the
overlap with the training half is shared by every predictor).
"""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_counts():
    z = np.load(os.path.join(HERE, "rewards.npz"), allow_pickle=True)
    ok = (z["reward"] > 0).astype(int)
    n_prompt = z["index"].max() + 1
    train = np.zeros(n_prompt, dtype=int)
    test = np.zeros(n_prompt, dtype=int)
    tot = np.zeros(n_prompt, dtype=int)
    first_env = np.zeros(n_prompt, dtype=int)
    for i, s, r, e in zip(z["index"], z["sample_id"], ok, z["env"]):
        tot[i] += r
        if s < 4:
            train[i] += r
        else:
            test[i] += r
        first_env[i] = e
    return train, test, tot, first_env, list(z["env_names"])


def probit_fit(counts, m_tr):
    """(m, tau) by marginal ML of counts ~ Bin(m_tr, Phi(theta))."""
    zq, wq = np.polynomial.hermite_e.hermegauss(61)
    wq = wq / wq.sum()
    s_vals, s_cnts = np.unique(counts, return_counts=True)
    logC = (gammaln(m_tr + 1) - gammaln(s_vals + 1)
            - gammaln(m_tr - s_vals + 1))

    def nll(pars):
        m, log_tau = pars
        th = m + np.exp(log_tau) * zq
        lp = norm.logcdf(th)
        ls = norm.logcdf(-th)
        ll_sz = (logC[:, None] + s_vals[:, None] * lp[None, :]
                 + (m_tr - s_vals)[:, None] * ls[None, :])
        return -(s_cnts * logsumexp(ll_sz + np.log(wq)[None, :],
                                    axis=1)).sum()

    best = None
    for m0 in (-2.0, -1.0, 0.0):
        r = minimize(nll, np.array([m0, 0.0]), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    m, tau = best.x[0], float(np.exp(best.x[1]))
    return m, tau, zq, wq


def probit_predict(strain, m_tr, m, tau, zq, wq, k):
    """E[1-(1-p)^k | s_train] per prompt, quadrature posterior."""
    th = m + tau * zq
    p = norm.cdf(th)
    loglik = (strain[:, None] * norm.logcdf(th)[None, :]
              + (m_tr - strain)[:, None] * norm.logcdf(-th)[None, :])
    w = np.exp(loglik - logsumexp(loglik + np.log(wq)[None, :],
                                  axis=1, keepdims=True)) * wq[None, :]
    return (w * (1.0 - (1.0 - p[None, :]) ** k)).sum(1)


def betabin_fit(counts, m_tr):
    s_vals, s_cnts = np.unique(counts, return_counts=True)
    logC = (gammaln(m_tr + 1) - gammaln(s_vals + 1)
            - gammaln(m_tr - s_vals + 1))

    def nll(pars):
        a, b = np.exp(pars)
        ll = (logC + betaln(s_vals + a, m_tr - s_vals + b)
              - betaln(a, b))
        return -(s_cnts * ll).sum()

    best = None
    for x0 in ((0.0, 0.0), (-1.0, 1.0)):
        r = minimize(nll, np.array(x0), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    return tuple(np.exp(best.x))


def betabin_predict(strain, m_tr, a, b, k):
    """E[1-(1-p)^k | s]: posterior Beta(a+s, b+m-s); use the falling
    product identity E[(1-p)^k] = prod_{j<k} (b'+j)/(a'+b'+j)."""
    ap = a + strain
    bp = b + m_tr - strain
    e = np.ones_like(ap, dtype=float)
    for j in range(k):
        e *= (bp + j) / (ap + bp + j)
    return 1.0 - e


def pass_at_k_unbiased(tot, n, k):
    """mean_i [1 - C(n - s_i, k)/C(n, k)]"""
    s = tot
    num = (gammaln(n - s + 1) - gammaln(k + 1) - gammaln(n - s - k + 1))
    den = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    frac = np.where(n - s >= k, np.exp(num - den), 0.0)
    return float((1.0 - frac).mean())


if __name__ == "__main__":
    train, test, tot, env, env_names = load_counts()
    n = len(train)
    y = (test > 0).astype(float)          # held-out pass@4, Bernoulli
    print(f"{n} prompts; base rate held-out pass@4 = {y.mean():.4f}; "
          f"train success mean {train.mean()/4:.4f}")

    p_plug = 1.0 - (1.0 - train / 4.0) ** 4
    m, tau, zq, wq = probit_fit(train, 4)
    p_probit = probit_predict(train, 4, m, tau, zq, wq, 4)
    a, b = betabin_fit(train, 4)
    p_bb = betabin_predict(train, 4, a, b, 4)
    print(f"probit difficulty prior: m={m:.3f} tau={tau:.3f} | "
          f"beta prior: a={a:.3f} b={b:.3f}")

    results = dict(n_prompts=int(n), probit=dict(m=m, tau=tau),
                   betabin=dict(a=a, b=b))
    for name, p in (("plugin", p_plug), ("probit", p_probit),
                    ("betabin", p_bb)):
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
        br = ((p - y) ** 2).mean()
        results[f"score_{name}"] = dict(logloss=float(ll),
                                        brier=float(br))
        print(f"  {name:8s} held-out pass@4: log loss {ll:.4f}  "
              f"Brier {br:.4f}")

    # aggregate extrapolation: mean pass@k, k = 1..8, from train half
    agg = {}
    for k in (1, 2, 4, 8):
        truth = pass_at_k_unbiased(tot, 8, k)
        pred_plug = float((1 - (1 - train / 4.0) ** k).mean())
        pred_probit = float(probit_predict(train, 4, m, tau, zq, wq,
                                           k).mean())
        pred_bb = float(betabin_predict(train, 4, a, b, k).mean())
        agg[k] = dict(truth=truth, plugin=pred_plug,
                      probit=pred_probit, betabin=pred_bb)
        print(f"  aggregate pass@{k}: truth {truth:.4f} | plugin "
              f"{pred_plug:.4f} | probit {pred_probit:.4f} | "
              f"betabin {pred_bb:.4f}")
    results["aggregate"] = agg

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")
