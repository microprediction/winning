"""Three-outcome margin stopping on the same replayed pairs.

The binary replay measured the failure: with gaps at or under one
point and a thousand shared items, early directional stops carry
30-40 percent realized error for every rule. The repair is a third
outcome. With an indifference margin delta, the rule watches the
posterior of the paired accuracy gap Delta = pi10 - pi01 and stops
on whichever of three events first reaches confidence 1 - alpha:

  A-better   P(Delta >  delta) >= 1 - alpha
  B-better   P(Delta < -delta) >= 1 - alpha
  tied       P(|Delta| <= delta) >= 1 - alpha

This is indifference-zone selection, and its posterior quantity is a
margin probability: the two-model special case of the winner-margin
spectrum m_i(delta) computed by the shifted survival field in the
general race. Posterior: Dirichlet(1,1,1,1) over the four pair
categories, with the gap's posterior treated as Gaussian (means,
variances, and the negative covariance of the two discordant
categories from the Dirichlet), accurate at these counts.

Truth per pair from the full log: direction if |Delta_full| > delta,
tied if |Delta_full| <= delta. An error is any verdict inconsistent
with that truth; running out of items is 'undecided', not an error.
"""
import glob
import json
import os

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
R_ORDERINGS = 400
ALPHA = 0.05
SEED = 13


def replay(a, b, delta):
    n = len(a)
    d_full = a.mean() - b.mean()
    truth = ("A" if d_full > delta else
             "B" if d_full < -delta else "tied")
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(R_ORDERINGS):
        order = rng.permutation(n)
        aa, bb = a[order].astype(np.int64), b[order].astype(np.int64)
        n10 = np.cumsum((aa == 1) & (bb == 0))
        n01 = np.cumsum((aa == 0) & (bb == 1))
        t = np.arange(1, n + 1)
        tot = t + 4.0
        m10 = (n10 + 1) / tot
        m01 = (n01 + 1) / tot
        v10 = m10 * (1 - m10) / (tot + 1)
        v01 = m01 * (1 - m01) / (tot + 1)
        cov = -m10 * m01 / (tot + 1)
        mu = m10 - m01
        sd = np.sqrt(v10 + v01 - 2 * cov)
        pA = 1.0 - ndtr((delta - mu) / sd)
        pB = ndtr((-delta - mu) / sd)
        pT = ndtr((delta - mu) / sd) - ndtr((-delta - mu) / sd)
        conf = 1 - ALPHA
        hitA = np.flatnonzero(pA >= conf)
        hitB = np.flatnonzero(pB >= conf)
        hitT = np.flatnonzero(pT >= conf)
        firsts = [(hitA[0] if len(hitA) else n, "A"),
                  (hitB[0] if len(hitB) else n, "B"),
                  (hitT[0] if len(hitT) else n, "tied")]
        stop, verdict = min(firsts)
        if stop == n:
            verdict, stop = "undecided", n - 1
        rows.append((stop + 1, verdict))
    items = np.array([r[0] for r in rows], dtype=float)
    verdicts = [r[1] for r in rows]
    counts = {v: verdicts.count(v) / len(verdicts)
              for v in ("A", "B", "tied", "undecided")}
    err = np.mean([v not in (truth, "undecided") for v in verdicts])
    return dict(truth=truth, mean_items=float(items.mean()),
                outcomes=counts, error=float(err))


if __name__ == "__main__":
    results = {}
    for path in sorted(glob.glob(os.path.join(HERE, "pair_*.npz"))):
        z = np.load(path)
        a, b = z["a"], z["b"]
        names = [str(x) for x in z["names"]]
        tag = f"{names[0]}_vs_{names[1]}"
        results[tag] = {}
        gap = abs(a.mean() - b.mean()) * 100
        print(f"[{tag}] gap {gap:.1f}pt")
        for delta in (0.01, 0.02, 0.03):
            r = replay(a, b, delta)
            results[tag][f"delta_{delta}"] = r
            oc = r["outcomes"]
            print(f"  delta={delta:.2f} truth={r['truth']:5s} "
                  f"items {r['mean_items']:6.1f}  "
                  f"A {oc['A']:.2f} B {oc['B']:.2f} "
                  f"tied {oc['tied']:.2f} undec {oc['undecided']:.2f}"
                  f"  error {r['error']:.3f}")
    with open(os.path.join(HERE, "results_margin.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results_margin.json")
