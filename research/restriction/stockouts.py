"""Restricted-menu shares from real stockouts.

Every other collection in this study either derives subset choice from a ranking or
comes from a laboratory experiment. This one is neither: in a grocery network,
products physically sell out, and shoppers arriving afterwards face a smaller menu
that nobody chose for the purposes of an experiment.

FreshRetailNet-50K (Dingdong Inc., CC BY 4.0) gives, for each store, product and day,
hourly sales and an hourly stock-status flag. The flag is 1 when the product is out;
sales in flagged hours average 0.005 against 0.055 in unflagged hours, an elevenfold
drop, which confirms the polarity and also shows the flag is imperfect, since a
stockout beginning mid-hour leaves some sales behind it.

The menu is a store crossed with a third-level product category. An occasion is a
store-category-date-hour. Occasions where every product in the category is in stock
give the full-menu shares; occasions with exactly one product out give the restricted
menus to be predicted. Both maps are calibrated on full-stock occasions from training
dates only and neither sees a restricted occasion. Sales quantities act as choice
weights, so the score is a weighted log loss.

Two caveats belong with any result. Stockouts follow demand, so restricted occasions
are not exchangeable with full-stock ones; the hour-of-day conditioning below is a
partial answer and the fitted-Luce null, run on the observed stockout pattern, is the
rest. And the flag's mid-hour latency means a nominally full menu may have been short
for part of the hour.

Usage:  python stockouts.py [max_rows] [min_products]
"""
import collections
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

FLOOR = 1e-6
ALPHA = 0.5
CACHE = HERE / "data" / "stockout_cells.npz"


def harvest(max_rows, min_products):
    """Stream the dataset once and keep per-occasion sales and in-stock sets."""
    from datasets import load_dataset
    d = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train", streaming=True)
    cells = collections.defaultdict(dict)          # (store,cat) -> (date,hour) -> dict
    for r in itertools.islice(iter(d), max_rows):
        st = r["hours_stock_status"]
        sl = r["hours_sale"]
        if len(st) != 24 or len(sl) != 24:
            continue
        key = (r["store_id"], r["third_category_id"])
        pid = r["product_id"]
        for h in range(24):
            occ = cells[key].setdefault((r["dt"], h), {})
            occ[pid] = (int(st[h]), float(sl[h]))
    keep = {k: v for k, v in cells.items()
            if len({p for occ in v.values() for p in occ}) >= min_products}
    return keep


def weighted_loss(q, w):
    q = np.maximum(np.asarray(q, dtype=float), FLOOR)
    q = q / q.sum()
    w = np.asarray(w, dtype=float)
    return float(-(w * np.log(q)).sum()), float(w.sum())


def run_cell(occs, folds=5, seed=0):
    """Held-out weighted log loss for the two maps on one store-category."""
    prods = sorted({p for occ in occs.values() for p in occ})
    K = len(prods)
    pi = {p: i for i, p in enumerate(prods)}
    dates = sorted({d for d, _ in occs})
    if len(dates) < folds or K < 3:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dates))
    fold = {dates[j]: f for f, blk in enumerate(np.array_split(order, folds)) for j in blk}
    tot_l = tot_g = tot_w = 0.0
    nocc = 0
    for f in range(folds):
        cts = np.zeros(K)
        for (dt, h), occ in occs.items():
            if fold[dt] == f or len(occ) < K:
                continue
            if any(occ[p][0] == 1 for p in prods if p in occ):
                continue                              # not a full menu
            for p in prods:
                cts[pi[p]] += occ[p][1]
        if cts.sum() <= 0:
            continue
        p_hat = (cts + ALPHA) / (cts.sum() + ALPHA * K)
        a, err = calibrate_np(list(p_hat))
        if err > 0.05:
            continue
        a = np.asarray(a)
        for (dt, h), occ in occs.items():
            if fold[dt] != f or len(occ) < K:
                continue
            live = [p for p in prods if occ[p][0] == 0]
            if len(live) < 2 or len(live) == K:
                continue                              # need a genuine restriction
            w = np.array([occ[p][1] for p in live])
            if w.sum() <= 0:
                continue
            idx = [pi[p] for p in live]
            lu = p_hat[idx] / p_hat[idx].sum()
            wp = win_probs_np(a[idx])
            ll, sw = weighted_loss(lu, w)
            gl, _ = weighted_loss(wp / wp.sum(), w)
            tot_l += ll
            tot_g += gl
            tot_w += sw
            nocc += 1
    if tot_w <= 0 or nocc < 20:
        return None
    return {"K": K, "occasions": nocc, "weight": tot_w,
            "luce": tot_l / tot_w, "race": tot_g / tot_w,
            "gain": (tot_l - tot_g) / tot_w}


def luce_null_cell(occs, reps, seed=7):
    """Redraw the sales weights from an exact Luce process on the observed menus.

    The stockout pattern, the hours and the total quantity per occasion are held
    exactly as observed; only the split across surviving products is regenerated. So
    any gain here is what contraction earns from estimation and from the stockout
    pattern itself, with no departure from the axiom anywhere."""
    prods = sorted({p for occ in occs.values() for p in occ})
    K = len(prods)
    cts = np.zeros(K)
    for occ in occs.values():
        if len(occ) == K and all(occ[p][0] == 0 for p in prods):
            for i, p in enumerate(prods):
                cts[i] += occ[p][1]
    if cts.sum() <= 0:
        return None
    u = (cts + ALPHA) / (cts.sum() + ALPHA * K)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        syn = {}
        for k, occ in occs.items():
            tot = sum(v[1] for v in occ.values())
            live = [p for p in prods if occ.get(p, (1, 0.0))[0] == 0]
            if not live or tot <= 0:
                syn[k] = {p: (occ[p][0], 0.0) for p in occ}
                continue
            q = np.array([u[prods.index(p)] for p in live])
            q = q / q.sum()
            draw = rng.multinomial(max(int(round(tot * 10)), 1), q) / 10.0
            d = {p: (occ[p][0], 0.0) for p in occ}
            for p, w in zip(live, draw):
                d[p] = (occ[p][0], float(w))
            syn[k] = d
        r = run_cell(syn)
        if r:
            out.append(r["gain"])
    return out or None


def main():
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    min_products = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cells = harvest(max_rows, min_products)
    print(f"{len(cells)} store-category cells with at least {min_products} products",
          flush=True)
    res = []
    for k, occs in cells.items():
        r = run_cell(occs)
        if r:
            r["cell"] = k
            res.append(r)
    if not res:
        print("no scorable cells")
        return
    gains = np.array([r["gain"] for r in res])
    wts = np.array([r["weight"] for r in res])
    print(f"\nscorable cells: {len(res)}")
    print(f"  race lower loss in {(gains > 0).sum()} of {len(gains)}")
    print(f"  unweighted mean gain {gains.mean():+.4f}   median {np.median(gains):+.4f}")
    print(f"  sales-weighted mean  {float((gains*wts).sum()/wts.sum()):+.4f}")
    rng = np.random.default_rng(3)
    bs = sorted(float(gains[rng.integers(0, len(gains), len(gains))].mean())
                for _ in range(4000))
    print(f"  cell bootstrap 95%   [{bs[100]:+.4f}, {bs[3900]:+.4f}]")
    print("\n  fitted-Luce null on the observed stockout pattern:")
    nulls = []
    for r in res[:40]:
        nn = luce_null_cell(cells[r["cell"]], 8)
        if nn:
            nulls.append((r["gain"], float(np.median(nn))))
    if nulls:
        obs = np.array([a for a, _ in nulls]); nul = np.array([b for _, b in nulls])
        print(f"    {len(nulls)} cells   observed mean {obs.mean():+.4f}   "
              f"null mean {nul.mean():+.4f}   excess {float((obs-nul).mean()):+.4f}")
        print(f"    excess positive in {(obs>nul).sum()} of {len(nulls)} cells")


if __name__ == "__main__":
    main()
