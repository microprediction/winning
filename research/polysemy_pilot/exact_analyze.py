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

def win_probs_np(a):
    """P(X_i = max), X_i ~ N(a_i, 1). Adaptive grid covering all locations."""
    a = np.asarray(a, float)
    lo, hi = a.min() - 9.0, a.max() + 9.0
    X = np.linspace(lo, hi, max(2401, int((hi - lo) / 0.01)))
    DX = X[1] - X[0]
    z = X[None, :] - a[:, None]                     # (n, grid)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    from scipy_free_cdf import Phi_np
    cdf = Phi_np(z)
    logcdf = np.log(np.clip(cdf, 1e-300, 1.0))
    tot = logcdf.sum(axis=0, keepdims=True)
    others = np.exp(tot - logcdf)                    # prod of cdfs excluding i
    p = (pdf * others).sum(axis=1) * DX
    return p / p.sum()


def calibrate_np(target, iters=4000, tol=1e-3):
    t = np.log(np.clip(np.asarray(target, float), 1e-9, 1.0))
    a = np.zeros(len(target))
    lr = 0.5
    prev_err = np.inf
    for k in range(iters):
        p = np.log(np.clip(win_probs_np(a), 1e-12, 1.0))
        err = float(np.abs(t - p).max())
        if err < tol:
            break
        if err > prev_err:            # oscillating: damp
            lr *= 0.7
        prev_err = err
        a += lr * (t - p)
        a -= a.mean()
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
INVENTORY = {
    "color": "red blue green purple orange yellow pink black white teal turquoise magenta gray brown violet indigo".split(),
    "fruit": "mango banana pineapple papaya kiwi coconut apple strawberry watermelon peach pear plum cherry grape orange blueberry raspberry".split(),
    "animal": "elephant lion giraffe zebra cheetah hippo rhino dog cat horse rabbit hamster dolphin tiger wolf owl fox panda penguin koala otter bear monkey".split(),
    "musical instrument": "guitar violin cello harp banjo piano drums flute trumpet saxophone clarinet ukulele".split(),
    "planet": "mercury venus earth mars jupiter saturn uranus neptune pluto".split(),
    "metal": "gold silver platinum copper iron titanium steel aluminum tungsten".split(),
    "bird": "eagle hawk owl falcon robin sparrow cardinal hummingbird penguin parrot crow raven swan flamingo".split(),
    "flower": "tulip daffodil lily daisy crocus rose orchid sunflower peony lavender hydrangea".split(),
    "vegetable": "carrot potato beet radish turnip broccoli spinach kale tomato cucumber pepper onion corn asparagus".split(),
    "tree": "pine cedar spruce fir oak maple willow birch aspen redwood cherry magnolia".split(),
    "sport": "soccer basketball football baseball hockey volleyball tennis golf swimming running cricket rugby badminton".split(),
    "hot drink": "coffee tea cocoa chai matcha cider".split(),
    "month": "january february march april may june july august september october november december".split(),
    "day of the week": "monday tuesday wednesday thursday friday saturday sunday".split(),
    "letter of the alphabet": list("abcdefghijklmnopqrstuvwxyz"),
    "state in the u.s.": "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington wisconsin wyoming".split(),
    "gemstone": "sapphire topaz aquamarine turquoise ruby emerald diamond opal amethyst garnet pearl jade".split(),
}


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
