"""Many rank markets at once, at golf-major scale: n = 150, quoting
win, top-5, top-10, top-20 and top-70 (the cut).

Five markets give 5(n-1) constraints against the 2n-2 free parameters
of (mu, log sigma) -- overdetermined threefold, so exact model-generated
quotes must still fit to solver precision (a consistency check on the
whole pipeline), and under quote noise the extra markets act as
replication: the fit should recover the true parameters better than the
dimension-matching two-market fit does.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "exp2_variances"))
from run_variances import gn_fit, align  # noqa: E402
from winning.factor.topk import top_k_probabilities  # noqa: E402


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n = 150
    ks = [1, 5, 10, 20, 70]
    mu_star = rng.normal(0, 0.8, n)
    mu_star -= mu_star.mean()
    s_star = rng.normal(0, 0.3, n)
    s_star -= s_star.mean()
    D_star = np.exp(2 * s_star)
    truth = {k: top_k_probabilities(mu_star, k, D=D_star) for k in ks}
    quoted_extra = {k: top_k_probabilities(mu_star, k, D=D_star)
                    for k in (3, 40)}
    out = {}

    # --- five exact markets -------------------------------------------
    t0 = time.time()
    (mu, s, sse, it), _ = gn_fit(truth, n, max_iter=40)
    mu_a, s_a = align(mu, s, mu_star, s_star)
    wall = time.time() - t0
    imp = {k: float(np.abs(top_k_probabilities(mu, k, D=np.exp(2 * s))
                           - quoted_extra[k]).max()) for k in (3, 40)}
    print(f"five exact markets, n=150: SSE {sse:.2e} in {it} iters "
          f"({wall:.0f} s); mu err {np.abs(mu_a-mu_star).max():.2e}, "
          f"log sigma err {np.abs(s_a-s_star).max():.2e}")
    print(f"  implied unquoted top-3/top-40 vs truth: "
          f"{ {k: f'{v:.1e}' for k, v in imp.items()} }")
    out["five_exact"] = dict(sse=sse, iters=it, wall=round(wall, 1),
                             mu_err=float(np.abs(mu_a - mu_star).max()),
                             s_err=float(np.abs(s_a - s_star).max()),
                             implied=imp)

    # --- noisy: five markets vs the dimension-matching two -----------
    noise = 0.05
    noisy = {}
    for k in ks:
        e = rng.normal(0, noise, n)
        qn = truth[k] * np.exp(e - e.mean())
        noisy[k] = qn * (k / qn.sum())

    for label, tgt in (("two", {1: noisy[1], 70: noisy[70]}),
                       ("five", noisy)):
        t0 = time.time()
        (muf, sf, ssef, itf), _ = gn_fit(tgt, n, max_iter=40)
        mua, sa = align(muf, sf, mu_star, s_star)
        wall = time.time() - t0
        print(f"{label} noisy markets: SSE {ssef:.3f} in {itf} iters "
              f"({wall:.0f} s); mu err {np.abs(mua-mu_star).max():.3f}, "
              f"log sigma err {np.abs(sa-s_star).max():.3f} "
              f"(noise {noise})")
        out[f"{label}_noisy"] = dict(
            sse=ssef, iters=itf, wall=round(wall, 1),
            mu_err=float(np.abs(mua - mu_star).max()),
            s_err=float(np.abs(sa - s_star).max()))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
