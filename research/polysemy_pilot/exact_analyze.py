"""Analyze exact_raw.json: Luce vs Thurstone zero-parameter predictions of
qualified distributions. Vectorized numpy Thurstone (grid integration)."""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import CELLS, MODELS, PHRASINGS, match_items

# Both functions delegate to winning.factor.core, the package's own adaptive
# log-domain lattice (points sized to the field, tails to 8 sigma, tol=1e-9 Newton-style
# inversion). Round-trip error on well-conditioned targets is 1e-10 to 1e-11, under ten
# milliseconds; the fixed-grid solver this replaced floored at 1e-3 to 1e-4 and a naive
# tightened version of it still only reached 1e-8. winning.factor.core is min-wins
# internally, so locations and scores are negated at the boundary to keep the max-wins
# convention (highest draw wins) this research code and the paper both use.
# `winning.factor.core` lives in the repository's top-level winning/ tree, which is
# not the package `pip install winning` gives (that resolves to src/winning, the
# ratings-layer renovation, and has no factor module). Insert the repo root ahead of
# site-packages so this always finds the right one regardless of the caller's cwd.
_REPO_ROOT = HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from winning.factor.core import win_probabilities as _win_probabilities_min
from winning.factor.core import abilities_from_probabilities as _abilities_min


def win_probs_np(a):
    """P(X_i = max), X_i ~ N(a_i, 1) iid."""
    return _win_probabilities_min(-np.asarray(a, dtype=float))


def calibrate_np(target, iters=500, tol=1e-9):
    """Invert shares to locations under the max-wins Gaussian race.

    Returns (a, err) as before: locations with mean zero, and the maximum absolute
    log-share residual, so every existing caller and its err > 0.05 acceptance gate
    is unchanged.
    """
    p = np.asarray(target, dtype=float)
    p = p / p.sum()
    a = -_abilities_min(p, n_iter=iters, tol=tol)
    model = np.maximum(win_probs_np(a), 1e-15)
    err = float(np.abs(np.log(model) - np.log(np.maximum(p, 1e-300))).max())
    return a, err


def rmse(pred, act):
    pred, act = np.asarray(pred), np.asarray(act)
    return float(np.sqrt(((pred - act) ** 2).mean()))


def entropy_norm(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    if len(p) < 2:
        return 0.0
    return float(-(p * np.log2(p)).sum() / math.log2(len(p)))


# Full per-category inventories: the Thurstone calibration field is the
# inventory intersected with matched tokens, mirroring the full state list
# of Cotton (2024). Keyed by unqualified noun.
# 17-category base set, frozen; see inventory.py
from inventory import BASE_INVENTORY as INVENTORY


def main():
    raw = json.loads((HERE / "exact_raw.json").read_text())
    results = []
    for unq_noun, q_noun, subset in CELLS:
        subset_l = [s.lower() for s in subset]
        inventory = set(INVENTORY[unq_noun.lower()])
        for ph in PHRASINGS:
            for m in MODELS:
                ku = f"{m}||{ph.format(c=unq_noun)}"
                kq = f"{m}||{ph.format(c=q_noun)}"
                if ku not in raw or kq not in raw:
                    continue
                m_unq = {k: v for k, v in match_items(raw[ku]).items()
                         if k in inventory}
                m_q = {k: v for k, v in match_items(raw[kq]).items()
                       if k in inventory}
                common = [s for s in subset_l if s in m_unq and s in m_q]
                if len(common) < 2:
                    continue

                az = sum(m_q[s] for s in common)
                actual = [m_q[s] / az for s in common]

                items = sorted(m_unq, key=m_unq.get, reverse=True)
                uz = sum(m_unq.values())
                p_full = [m_unq[s] / uz for s in items]

                lz = sum(m_unq[s] for s in common)
                luce_pred = [m_unq[s] / lz for s in common]

                a, cal_err = calibrate_np(p_full)
                if cal_err > 0.05:
                    print(f'skip (cal_err={cal_err:.3f}): {m} {unq_noun}->{q_noun}')
                    continue
                idx = [items.index(s) for s in common]
                thur_pred = win_probs_np(a[idx])
                thur_pred = (thur_pred / thur_pred.sum()).tolist()

                results.append({
                    "cell": f"{unq_noun} -> {q_noun}", "phrasing": ph, "model": m,
                    "subset": common, "actual": actual,
                    "luce": luce_pred, "thurstone": thur_pred,
                    "rmse_luce": rmse(luce_pred, actual),
                    "rmse_thurstone": rmse(thur_pred, actual),
                    "H_unq": entropy_norm(p_full),
                    "matched_mass_unq": uz, "matched_mass_q": sum(m_q.values()),
                })

    (HERE / "exact_results.json").write_text(json.dumps(results, indent=1))
    print(f"{len(results)} usable cells")
    for m in MODELS:
        rs = [r for r in results if r["model"] == m]
        if not rs:
            continue
        tw = sum(r["rmse_thurstone"] < r["rmse_luce"] for r in rs)
        pl = rmse([x for r in rs for x in r["luce"]],
                  [x for r in rs for x in r["actual"]])
        pt = rmse([x for r in rs for x in r["thurstone"]],
                  [x for r in rs for x in r["actual"]])
        print(f"{m:<12} cells={len(rs):>3}  Thurstone wins {tw}/{len(rs)}  "
              f"pooled RMSE: Luce={pl:.4f} Thurstone={pt:.4f}")

    diffs = [r["rmse_luce"] - r["rmse_thurstone"] for r in results]
    n = len(diffs)
    random.seed(1234)
    B = 10000
    wins = sum(sum(diffs[random.randrange(n)] for _ in range(n)) > 0
               for _ in range(B))
    print(f"\nmean RMSE difference (Luce - Thurstone): {sum(diffs)/n:+.4f}")
    print(f"bootstrap P(Thurstone better) = {wins/B:.4f}  (n={n} cells)")

    # stratify by unqualified entropy (the regime effect from the 2024 paper)
    med = sorted(r["H_unq"] for r in results)[n // 2]
    for name, sel in [("low-H cells", lambda r: r["H_unq"] <= med),
                      ("high-H cells", lambda r: r["H_unq"] > med)]:
        rs = [r for r in results if sel(r)]
        tw = sum(r["rmse_thurstone"] < r["rmse_luce"] for r in rs)
        d = sum(r["rmse_luce"] - r["rmse_thurstone"] for r in rs) / len(rs)
        print(f"{name}: Thurstone wins {tw}/{len(rs)}, mean RMSE diff {d:+.4f}")


if __name__ == "__main__":
    main()
