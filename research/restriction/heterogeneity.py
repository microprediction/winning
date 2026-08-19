"""What the odds-discount ratio actually measures: population heterogeneity.

The ratio of the observed discount to the discount Case V predicts has an exact
null of one, verified at every noise scale. Across ten human datasets it runs from
0.78 to 3.49, and the pattern is not noise: it is small in idiosyncratic domains
(humour, perceptual discrimination, car attributes) and large in factional ones
(occupational prestige, political goals, the APA presidential ballot whose
electorate splits between research and clinical psychologists).

This simulates the obvious mechanism. Let every individual be a Case V contest,
so Thurstone holds exactly at the level of the person, and let the population be
a mixture of groups with different location vectors. A mixture of contests is not
a contest, so the aggregate departs from Case V even though no individual does.

Measured 2026-08-12:

    single homogeneous group            1.00     (the null, recovered)
    three factions                      2.51
    two factions                        3.99 to 4.47

    observed: humour 1.12, perception 1.16, leisure 1.14, car attributes 0.78,
              Rice ballot 0.89, Dublin West 1.26, sushi and Netflix 1.33,
              occupations 1.97, political goals 2.35, APA ballot 3.49

So the ratio is a heterogeneity index rather than a verdict on Thurstone, and the
human data are consistent with Thurstone holding for individuals while aggregation
supplies the apparent violation.

The machine number resists that explanation and is the more interesting for it.
Machines sit at 5.77, above every faction simulated here, while their output
distribution is markedly LESS diverse than a human population's, contesting two of
ten alternatives where people contest nine. Heterogeneity cannot be producing a
ratio in a population that is more homogeneous than the comparison, so whatever
drives the machine violation is not the mechanism that explains the human one.

Usage:  python heterogeneity.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np


def ratio_for(locs, weights, K=5, N=400000, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    who = rng.choice(len(weights), size=N, p=weights)
    X = np.array(locs)[who] + sigma * rng.standard_normal((N, K))
    full = np.bincount(X.argmax(1), minlength=K) / N
    live = [i for i in range(K) if full[i] > 1e-4]
    p = [full[i] for i in live]
    z = sum(p)
    a_hat, _ = calibrate_np([x / z for x in p])
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pr = np.bincount(X[:, [i, j]].argmax(1), minlength=2) / N
            if min(pr) <= 0:
                continue
            L = math.log(full[i] / full[j])
            if abs(L) < 1e-6:
                continue
            obs.append((L, math.log(pr[0] / pr[1]) - L))
            w = win_probs_np(a_hat[[x, y]])
            if w[0] > 0 and w[1] > 0:
                cv.append((L, math.log(w[0] / w[1]) - L))
    lam = lambda P: sum(-d * L for L, d in P) / sum(L * L for L, _ in P)
    return lam(obs) / lam(cv)


def main():
    base = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(f"{'population':<44}{'ratio':>7}")
    print(f"{'single homogeneous group (the null)':<44}"
          f"{ratio_for([base], [1.0]):>7.2f}")
    for gap, lab in ((1.0, "two factions, mild disagreement"),
                     (2.5, "two factions, strong disagreement"),
                     (5.0, "two factions, opposed orderings")):
        A = [a * gap / 2 for a in base]
        B = [b * gap / 2 for b in base[::-1]]
        print(f"{lab:<44}{ratio_for([A, B], [0.5, 0.5]):>7.2f}")
    print(f"{'three factions, opposed':<44}"
          f"{ratio_for([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]],
                       [1 / 3, 1 / 3, 1 / 3]):>7.2f}")
    print("\nobserved human range 0.78 to 3.49; machines 5.77")


if __name__ == "__main__":
    main()
