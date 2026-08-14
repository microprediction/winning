"""Contraction slope per dataset, with intervals, from committed data only.

The draft's scorecard reported one point estimate per row and merged datasets that
should not have been merged: a "Sushi and Netflix" row combined Sushi's 45 pairs
with three Netflix files of three pairs each, two of which shift the opposite way.
It also gave no uncertainty, while three rows rest on six, ten or fourteen pairs.

This regenerates the table one dataset at a time. For each, using the full-field
first-place shares p and the observed share q_ij placing i above j:

    delta_ij = logit(q_ij) - log(p_i / p_j),      i the higher-share alternative,

fitted through the origin as delta = -lambda log(p_i/p_j) over unordered pairs. The
same statistic is computed for the Gaussian race's own predicted pairwise
probabilities, giving the value the race implies rather than an asserted zero, and
the residual is the difference. Intervals come from resampling respondents and
recomputing everything downstream: shares, calibration, race prediction and both
slopes. That is the referee's requirement, and it is what the single-fit numbers in
the draft could not supply.

Usage:  python lambda_table.py [n_boot]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all, ranks_from_ratings

DATA = HERE / "data"
PREFLIB = HERE / "preflib"
ALPHA = 0.5           # add-alpha, applied identically wherever shares are formed


def read_soc(path):
    """Complete rankings from a PrefLib order file; returns a ranks matrix."""
    lines = [l.strip() for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    rows = []
    for line in lines:
        # PrefLib order files are "count: a,b,c,d"; older files use a comma there
        head, sep, tail = line.partition(":")
        if not sep:
            head, _, tail = line.partition(",")
        try:
            cnt = int(head.strip())
            order = [int(x) for x in tail.replace("{", "").replace("}", "").split(",")
                     if x.strip()]
        except ValueError:
            continue
        if len(order) < 2 or len(set(order)) != len(order):
            continue
        rows.append((cnt, order))
    if not rows:
        return None
    K = max(max(o) for _, o in rows)
    full = [(c, o) for c, o in rows if len(o) == K]
    if not full:
        return None
    R = []
    for c, o in full:
        r = [0.0] * K
        for pos, it in enumerate(o):
            r[it - 1] = pos + 1
        R.extend([r] * min(c, 20000))
    return np.array(R, dtype=float)


def preflib_sets():
    out = {}
    for tag, label in (("dots", "Dots"), ("puzzle", "Puzzles"),
                       ("netflix", "Netflix")):
        mats = []
        for f in sorted(PREFLIB.glob(f"*{tag}*")):
            M = read_soc(f)
            if M is not None and M.shape[0] >= 50:
                mats.append(M)
        if not mats:
            continue
        # keep files separate when widths differ; otherwise stack
        widths = {m.shape[1] for m in mats}
        if len(widths) == 1:
            out[label] = np.vstack(mats)
        else:
            for i, m in enumerate(mats):
                out[f"{label} {i+1}"] = m
    return out


def slopes(R):
    """(observed lambda, race-implied lambda) for one ranks matrix."""
    n, K = R.shape
    cts = np.bincount(R.argmin(axis=1), minlength=K).astype(float)
    p = (cts + ALPHA) / (n + ALPHA * K)
    a, err = calibrate_np(list(p))
    if err > 0.05:
        return None
    num_o = num_r = den = 0.0
    npair = 0
    for i in range(K):
        for j in range(i + 1, K):
            hi, lo = (i, j) if p[i] >= p[j] else (j, i)
            L = np.log(p[hi] / p[lo])
            if not np.isfinite(L) or L <= 1e-12:
                continue
            # observed head-to-head share among respondents who ordered the pair
            better = (R[:, hi] < R[:, lo]).sum()
            q = (better + ALPHA) / (n + 2 * ALPHA)
            d_o = np.log(q / (1 - q)) - L
            w = win_probs_np(a[[hi, lo]])
            qr = float(np.clip(w[0] / w.sum(), 1e-9, 1 - 1e-9))
            d_r = np.log(qr / (1 - qr)) - L
            num_o += L * (-d_o)
            num_r += L * (-d_r)
            den += L * L
            npair += 1
    if den <= 0 or npair == 0:
        return None
    return num_o / den, num_r / den, npair


def run(name, R, nboot):
    base = slopes(R)
    if base is None:
        return None
    lo_, lr_, npair = base
    n = R.shape[0]
    rng = np.random.default_rng(3)
    bo, br, bd = [], [], []
    for _ in range(nboot):
        s = slopes(R[rng.integers(0, n, n)])
        if s is None:
            continue
        bo.append(s[0]); br.append(s[1]); bd.append(s[0] - s[1])
    if len(bo) < 20:
        return None
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return {"name": name, "n": n, "K": R.shape[1], "pairs": npair,
            "obs": lo_, "race": lr_, "resid": lo_ - lr_,
            "obs_ci": q(bo), "resid_ci": q(bd)}


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    data = load_all()
    data.update(preflib_sets())
    print(f"{'dataset':<20}{'n':>7}{'K':>3}{'pairs':>6}"
          f"{'observed lambda':>26}{'race':>8}{'residual (obs-race)':>26}")
    rows = []
    for name, R in sorted(data.items()):
        r = run(name, R, nboot)
        if r is None:
            print(f"{name:<20}  not computable")
            continue
        rows.append(r)
        print(f"{r['name']:<20}{r['n']:>7}{r['K']:>3}{r['pairs']:>6}"
              f"{r['obs']:>10.3f} [{r['obs_ci'][0]:+.3f},{r['obs_ci'][1]:+.3f}]"
              f"{r['race']:>8.3f}"
              f"{r['resid']:>10.3f} [{r['resid_ci'][0]:+.3f},{r['resid_ci'][1]:+.3f}]",
              flush=True)
    pos = [r for r in rows if r["obs_ci"][0] > 0]
    closer = [r for r in rows if abs(r["resid"]) < abs(r["obs"])]
    print(f"\ncontraction interval excludes zero in {len(pos)}/{len(rows)} datasets")
    print(f"race closer than renormalization (|residual| < |observed|) "
          f"in {len(closer)}/{len(rows)}")


if __name__ == "__main__":
    main()
