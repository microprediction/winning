"""Rebuilt pass@k study, actioning the eight-point review.

Changes from run_passk.py, each tied to a review finding:
1. OUTCOME DEFINITION: RLVE shapes rewards, so reward > 0 includes
   partial credit. Success is now reward >= 1 - 1e-6; the loose
   definition is kept as a sensitivity column. The extraction kept
   only (index, sample_id, reward, env), and the source rows carry
   no separate accuracy field, so the strict threshold is the
   available definition and is stated as such.
7. EXTRAPOLATION FOR REAL: train on samples 0-1 (m = 2), hold out
   samples 2-7. The held-out per-prompt pass@6 is one Bernoulli and
   k = 6 > m = 2 is genuine extrapolation with a disjoint target;
   the aggregate reference for k <= 6 is the combinatorial estimator
   on the six held-out samples only (disjoint, and labeled a
   reference estimate, not truth).
4. BASELINES AND CLIPPING: the raw plug-in is reported at four
   clipping levels (its unclipped log loss is infinite); added
   baselines: plug-in at the fitted posterior mean (separating
   smoothing from integration -- q_k is concave so integration
   LOWERS the prediction relative to the mean), and the fixed
   Jeffreys Beta(1/2,1/2) posterior predictive (no fitted
   hyperparameters).
5. KAZDAN ROW SEPARATE: the prior expectation under the fitted
   prior is its own row; the empirical average of posterior
   predictions is another; they differ on the fitted sample.
8. CALIBRATION BY COUNT and a THRESHOLD DECISION: predicted vs
   observed held-out success rate per training count, and the
   reachability decision (predicted chance of solving within 6
   attempts >= 0.05, the lower edge of TailSFT's band): the plug-in
   declares every zero-count prompt unreachable; the posterior does
   not; the held-out outcomes score the disagreement.
Also verified here: the reviewer's counterexample to
monotonicity-in-k of the Jensen bias (B_{2,3}(0.9) > B_{2,4}(0.9)),
and the reviewer's reproduction numbers on the old definition.
"""
import json
import os
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def load(strict=True):
    z = np.load(os.path.join(HERE, "rewards.npz"), allow_pickle=True)
    ok = (z["reward"] >= 1 - 1e-6 if strict
          else z["reward"] > 0).astype(int)
    n = z["index"].max() + 1
    m_train, s_train = 2, np.zeros(n, dtype=int)
    s_test = np.zeros(n, dtype=int)
    for i, sid, r in zip(z["index"], z["sample_id"], ok):
        if sid < m_train:
            s_train[i] += r
        else:
            s_test[i] += r
    return s_train, s_test, m_train, 6, z


def betabin_fit(counts, m):
    sv, sc = np.unique(counts, return_counts=True)
    logC = gammaln(m + 1) - gammaln(sv + 1) - gammaln(m - sv + 1)

    def nll(x):
        a, b = np.exp(x)
        return -(sc * (logC + betaln(sv + a, m - sv + b)
                       - betaln(a, b))).sum()
    best = min((minimize(nll, np.array(x0), method="Nelder-Mead")
                for x0 in ((0.0, 0.0), (-1.0, 1.0))),
               key=lambda r: r.fun)
    return tuple(np.exp(best.x))


def probit_fit(counts, m):
    zq, wq = np.polynomial.hermite_e.hermegauss(61)
    wq = wq / wq.sum()
    sv, sc = np.unique(counts, return_counts=True)
    logC = gammaln(m + 1) - gammaln(sv + 1) - gammaln(m - sv + 1)

    def nll(x):
        mu, lt = x
        th = mu + np.exp(lt) * zq
        ll = (logC[:, None] + sv[:, None] * norm.logcdf(th)[None, :]
              + (m - sv)[:, None] * norm.logcdf(-th)[None, :])
        return -(sc * logsumexp(ll + np.log(wq)[None, :], 1)).sum()
    best = min((minimize(nll, np.array([m0, 0.0]),
                         method="Nelder-Mead")
                for m0 in (-2.0, -1.0, 0.0)), key=lambda r: r.fun)
    return best.x[0], float(np.exp(best.x[1])), zq, wq


def beta_post_pred(s, m, a, b, k):
    ap, bp = a + s, b + m - s
    e = np.ones_like(ap, dtype=float)
    for j in range(k):
        e *= (bp + j) / (ap + bp + j)
    return 1.0 - e


def beta_post_mean(s, m, a, b):
    return (a + s) / (a + b + m)


def probit_post_pred(s, m, mu, tau, zq, wq, k):
    th = mu + tau * zq
    p = norm.cdf(th)
    ll = (s[:, None] * norm.logcdf(th)[None, :]
          + (m - s)[:, None] * norm.logcdf(-th)[None, :])
    w = np.exp(ll - logsumexp(ll + np.log(wq)[None, :], 1,
                              keepdims=True)) * wq[None, :]
    return (w * (1 - (1 - p[None, :]) ** k)).sum(1)


def scores(p, y, clip=1e-12):
    pc = np.clip(p, clip, 1 - clip)
    return (float(-(y * np.log(pc)
                    + (1 - y) * np.log(1 - pc)).mean()),
            float(((p - y) ** 2).mean()))


def ref_pass_at_k(s, n, k):
    num = gammaln(n - s + 1) - gammaln(k + 1) - gammaln(n - s - k + 1)
    den = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    frac = np.where(n - s >= k, np.exp(num - den), 0.0)
    return float((1 - frac).mean())


if __name__ == "__main__":
    out = {}

    # --- reviewer verification block, old definition and design ---
    z = np.load(os.path.join(HERE, "rewards.npz"), allow_pickle=True)
    r = z["reward"]
    out["reward_counts"] = dict(
        equal_one=int((np.abs(r - 1) < 1e-9).sum()),
        strictly_between=int(((r > 0) & (r < 1 - 1e-9)).sum()))
    print(f"[labels] rewards==1: {out['reward_counts']['equal_one']}, "
          f"in (0,1): {out['reward_counts']['strictly_between']}")

    # monotonicity counterexample B_{2,k}(0.9)
    def bias(m, k, p):
        s = np.arange(m + 1)
        pmf = (np.exp(gammaln(m + 1) - gammaln(s + 1)
                      - gammaln(m - s + 1))
               * p ** s * (1 - p) ** (m - s))
        return (1 - (1 - p) ** k) - float(
            (pmf * (1 - (1 - s / m) ** k)).sum())
    out["bias_counterexample"] = dict(B23=bias(2, 3, 0.9),
                                      B24=bias(2, 4, 0.9))
    print(f"[monotonicity counterexample] B_2,3(0.9)="
          f"{out['bias_counterexample']['B23']:.5f}  B_2,4(0.9)="
          f"{out['bias_counterexample']['B24']:.5f}")

    # --- main study, both outcome definitions ---
    for label, strict in (("strict", True), ("loose", False)):
        s_tr, s_te, m, kk, _ = load(strict)
        n = len(s_tr)
        y = (s_te > 0).astype(float)          # pass@6 held out
        a, b = betabin_fit(s_tr, m)
        mu, tau, zq, wq = probit_fit(s_tr, m)
        preds = {
            "plugin": 1 - (1 - s_tr / m) ** kk,
            "plugin_at_post_mean": 1 - (1 - beta_post_mean(
                s_tr, m, a, b)) ** kk,
            "jeffreys": beta_post_pred(s_tr, m, 0.5, 0.5, kk),
            "beta": beta_post_pred(s_tr, m, a, b, kk),
            "probit": probit_post_pred(s_tr, m, mu, tau, zq, wq, kk),
        }
        rows = {}
        for name, p in preds.items():
            ll, br = scores(p, y)
            rows[name] = dict(logloss=ll, brier=br)
        clip_table = {c: scores(preds["plugin"], y, clip=c)[0]
                      for c in (1e-12, 1e-6, 1e-3, 1e-2)}
        agg = {}
        for k in (1, 2, 4, 6):
            agg[k] = dict(
                reference=ref_pass_at_k(s_te, 6, k),
                plugin=float((1 - (1 - s_tr / m) ** k).mean()),
                beta_post_avg=float(beta_post_pred(
                    s_tr, m, a, b, k).mean()),
                probit_post_avg=float(probit_post_pred(
                    s_tr, m, mu, tau, zq, wq, k).mean()),
                kazdan_prior=float(1 - np.exp(
                    sum(np.log((b + j) / (a + b + j))
                        for j in range(k)))),
            )
        calib = {}
        for sv in range(m + 1):
            mask = s_tr == sv
            calib[sv] = dict(
                n=int(mask.sum()),
                observed=float(y[mask].mean()),
                beta_pred=float(preds["beta"][mask].mean()),
                plugin_pred=float(preds["plugin"][mask].mean()))
        # reachability decision at tau = 0.05
        dec = {}
        for name in ("plugin", "beta"):
            unreach = preds[name] < 0.05
            dec[name] = dict(
                declared_unreachable=int(unreach.sum()),
                of_which_succeed=int(y[unreach].sum()))
        out[label] = dict(prior_beta=[a, b],
                          prior_probit=[mu, tau],
                          perprompt=rows, clip_table={str(c): v for
                                                      c, v in
                                                      clip_table.items()},
                          aggregate=agg, calibration=calib,
                          decision=dec)
        print(f"[{label}] beta({a:.3f},{b:.3f}) "
              f"probit({mu:.3f},{tau:.3f})")
        for nm, rr in rows.items():
            print(f"  {nm:20s} logloss {rr['logloss']:.4f} "
                  f"brier {rr['brier']:.4f}")
        print("  clip sensitivity (plugin logloss): "
              + "  ".join(f"{c:g}:{v:.3f}"
                          for c, v in clip_table.items()))
        for k in (1, 2, 4, 6):
            a_ = agg[k]
            print(f"  agg k={k}: ref {a_['reference']:.4f} plugin "
                  f"{a_['plugin']:.4f} beta-avg "
                  f"{a_['beta_post_avg']:.4f} kazdan-prior "
                  f"{a_['kazdan_prior']:.4f}")
        for sv, cc in calib.items():
            print(f"  calib s={sv}: n={cc['n']} observed "
                  f"{cc['observed']:.3f} beta {cc['beta_pred']:.3f} "
                  f"plugin {cc['plugin_pred']:.3f}")
        for nm, dd in dec.items():
            print(f"  decision {nm}: {dd['declared_unreachable']} "
                  f"declared unreachable, "
                  f"{dd['of_which_succeed']} of them succeed")

    with open(os.path.join(HERE, "results2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results2.json")
