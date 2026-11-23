"""Own-slope preconditioning vs the unpreconditioned BLP-style map.

abilities_from_race steps mu_i <- mu_i - clip(alpha r_i / d_i) with
d_i = (dp_i/dmu_i)/p_i from the same field pass. Setting d_i = -1 gives the
Berry (1994) log-share contraction in min-wins form (exact for logit at
small shares). Everything else -- start, clip, recentering, tolerance -- is
held fixed, so the iteration counts isolate the preconditioner; the BLP map
is also run damped (alpha 0.5, 0.25), since undamped it can overshoot. Factor
grammar, rank 2, three regimes of performance scale. Seeded; run:
python run_blp_ablation.py
"""
import warnings
import numpy as np
from winning.factor.races import race_probabilities

warnings.simplefilter("ignore")

def invert(target, V, D, precondition, alpha=1.0, n_iter=500, tol=1e-8):
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    for it in range(n_iter + 1):            # it = updates applied so far
        phat, sl = race_probabilities(mu, V=V, D=D, return_slopes=True)
        r = np.log(np.maximum(phat, 1e-300)) - logt
        if np.abs(r).max() < tol or it == n_iter:
            return it, float(np.abs(r).max())
        d = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6) if precondition else -np.ones_like(r)
        lim = np.minimum(2.0, 10.0 * np.abs(r))
        mu = mu - np.clip(alpha * r / d, -lim, lim)
        mu -= mu.mean()

rng = np.random.default_rng(31)
n = 200
print(f"{'regime':34s} {'own-slope':>18s} {'BLP':>18s} {'BLP a=0.5':>18s} {'BLP a=0.25':>18s}")
for label, scale, hetero in (("unit scale, mild heterogeneity", 1.0, 0.5),
                             ("wide performance sd (x3)", 3.0, 0.5),
                             ("heterogeneous sd (D in 0.1..4)", 1.0, None)):
    mu0 = np.sort(rng.normal(size=n)) * 1.5
    V = rng.normal(size=(n, 2)) * 0.4 * scale
    D = (0.6 + hetero * rng.random(n)) * scale ** 2 if hetero is not None else 0.1 + 3.9 * rng.random(n)
    target = race_probabilities(mu0, V=V, D=D)
    a = invert(target, V, D, True)
    cells = [a] + [invert(target, V, D, False, alpha=al) for al in (1.0, 0.5, 0.25)]
    print(f"{label:34s} " + "  ".join(f"{c[0]:4d} it, res {c[1]:.0e}" for c in cells), flush=True)
