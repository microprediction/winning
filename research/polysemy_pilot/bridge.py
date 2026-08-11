"""Where on the Luce-Thurstone bridge does machine restriction sit?

The composite random-utility model has performance noise

    eta_i = Z_i - tau * G_i,     Z ~ N(0, sd^2),  G iid Gumbel,

so that conditional on the latent state the choice is softmax at temperature
tau, and marginally it is a hard contest with composite noise. Sending sd to
zero gives pure Luce; sending tau to zero gives Case V Thurstone. Everything
between is a one-parameter family of noise shapes, and because calibrating
locations to the unrestricted distribution absorbs any common scaling, what is
identifiable from a restriction experiment is the mix

    rho = sd / tau,

not the two scales separately. rho = 0 is the urn, rho = infinity is the race.

A second and independent degree of freedom is whether the restricted stage is
noisier than the unrestricted one. That is not a change of family but of scale
between the two measurements, and Section 5.7 of the paper found it to be the
dominant effect. The two are fitted together here: shape and stage inflation,
both cross-fitted over question types so that no category informs its own
prediction.

Usage:  python bridge.py [n_cells]
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from winning.kernels import gumbel_min_kernel, softmax_thurstone_kernel
from winning.lattice_calibration import (state_price_implied_ability,
                                         ability_implied_state_prices)
from datastore import write_json_atomic

UNIT = 0.1
# rho = sd/tau. 0 is pure Luce, large is pure Case V.
RHOS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, None]   # None = pure Gaussian
INFLATE = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0]


def kernel(rho, inflate=1.0):
    """Composite kernel at mix rho, with all scales multiplied by inflate."""
    if rho is None:
        from winning.kernels import gaussian_kernel
        return gaussian_kernel(UNIT, sd=1.0 * inflate)
    if rho == 0.0:
        return gumbel_min_kernel(UNIT, scale=1.0 * inflate)
    tau = 1.0 * inflate
    return softmax_thurstone_kernel(UNIT, sd=rho * tau, temperature=tau)


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def load(n_cells, seed=5):
    rows = json.loads((HERE / "sweep_results.json").read_text())
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n_cells:
            break
        dp = r.get("deleted_p")
        if dp is None or not (0.0 < dp < 1.0):
            continue
        luce = r["luce"]
        p_full = [x * (1.0 - dp) for x in luce] + [dp]
        z = sum(p_full)
        if z <= 0 or min(p_full) <= 0:
            continue
        out.append({"cat": r["category"], "p_full": [x / z for x in p_full],
                    "actual": r["actual"], "luce": luce,
                    "thur": r["thurstone"], "n_keep": len(luce)})
    return out


def predict(cell, rho, inflate):
    """Calibrate on the unrestricted field, then drop the deleted item."""
    try:
        k_cal = kernel(rho, 1.0)
        a = state_price_implied_ability(cell["p_full"], k_cal, UNIT)
        keep = a[:-1]                      # the deleted item is last by construction
        k_pred = kernel(rho, inflate)
        q = ability_implied_state_prices(keep, k_pred, UNIT)
        s = sum(q)
        if s <= 0 or any(x < 0 for x in q):
            return None
        return [x / s for x in q]
    except Exception:
        return None


def score(cells, rho, inflate):
    tot, n = 0.0, 0
    for c in cells:
        q = predict(c, rho, inflate)
        if q is None or len(q) != len(c["actual"]):
            continue
        tot += kl(c["actual"], q)
        n += 1
    return (tot / n, n) if n else (float("inf"), 0)


def main():
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    cells = load(n_cells)
    cats = sorted({c["cat"] for c in cells})
    print(f"{len(cells)} cells over {len(cats)} question types, "
          f"lattice unit {UNIT}")

    rng = random.Random(17)
    sh = cats[:]
    rng.shuffle(sh)
    assign = {c: i % 5 for i, c in enumerate(sh)}

    held, chosen = [], []
    per_cat = defaultdict(list)
    for f in range(5):
        tr = [c for c in cells if assign[c["cat"]] != f]
        te = [c for c in cells if assign[c["cat"]] == f]
        if not tr or not te:
            continue
        best, arg = float("inf"), None
        for rho in RHOS:
            for inf in INFLATE:
                v, n = score(tr, rho, inf)
                if n > 0.5 * len(tr) and v < best:
                    best, arg = v, (rho, inf)
        if arg is None:
            continue
        chosen.append(arg)
        for c in te:
            q = predict(c, *arg)
            if q is None or len(q) != len(c["actual"]):
                continue
            k = kl(c["actual"], q)
            held.append(k)
            per_cat[c["cat"]].append(k)
        print(f"  fold {f}: chose rho={arg[0]} inflate={arg[1]} "
              f"(train KL {best:.4f}, {len(te)} held out)", flush=True)

    if not held:
        print("no held-out predictions")
        return
    print(f"\nheld-out mean KL, composite bridge: {sum(held)/len(held):.4f} "
          f"over {len(held)} cells")
    print(f"  selected (rho, inflate) per fold: {chosen}")
    print()
    print("  for comparison, on the same battery:")
    print("    renormalization, no parameter     1.926")
    print("    Case V removal, no parameter      1.786")
    print("    uniform over survivors            1.089")
    print("    tilted renormalization, one gamma 0.803")
    print("    Case V at fitted noise ratio      0.794")
    write_json_atomic(HERE / "bridge_results.json",
                      {"held_out_mean_kl": sum(held) / len(held),
                       "n": len(held), "folds": [list(map(str, c)) for c in chosen]})


if __name__ == "__main__":
    main()
