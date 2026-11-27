"""Barrage for a candidate step rule (the "sign" rule) in the diagonal
inversion, against the shipped rule, on the same fields and targets.

Both rules are run by copies of the two shipped loops
(winning.factor.races.abilities_from_race for independent and factor
fields, _abilities_from_structure for blocks, trees and nested fields).
Only the step multiplier differs:

  shipped   independent and factor: 1 for n > 2, 0.7 at n = 2.
            blocks, tree, nested: 0.7, halved (floor 0.05) whenever the
            largest residual grows.
  sign      one multiplier per contestant, starting at 1: halved (floor
            0.05) when that contestant's residual changes sign between
            sweeps, multiplied by 1.5 (cap 1) when it does not.

The start, the preconditioner, the residual-proportional clip, the
recentring and the tolerance (1e-8 on the log-probability residual) are
the shipped ones. Each run is allowed CAP sweeps; "fail" counts runs
above the shipped cap (60 sweeps for independent and factor, 120 for
the others), "stuck" those above CAP.

Stages (argument 1):
  small   n = 2 to 20, six classes, four strengths of shared variation,
          four spreads of ability, SEEDS replicates, three targets:
          round trip, Dirichlet(1) and Dirichlet(0.3)
  bases   independent and factor rank 1 with the Gumbel base, same grid
  large   n = 50 to 100,000, six classes, four strengths, spread 1,
          three replicates, round trip and Dirichlet(1)

Fields are those of run_low_dimension_barrage.py except that clusters
have ten members for n >= 50. Seed for (stage s, class g, n, scale
index j, spread index k, replicate r): default_rng([2028, s, g, n, j, k, r]).

    nice -n 15 python run_sign_rule_barrage.py small [SEEDS] [WORKERS]

Prints one line per (class, n, scale) cell and a list of every run on
which the sign rule fails or needs more sweeps than the shipped rule
plus ten. Also writes one CSV row per run to sign_rule_<stage>.csv.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np

from winning.factor.races import race_probabilities
from winning.factor.structures import (Blocks, Nested, Tree,
                                       dispatch_probabilities,
                                       structure_variances)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "small"
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else (3 if STAGE == "large" else 20)
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
CLASSES = ("independent", "factor rank 1", "factor rank 2", "blocks",
           "tree", "nested")
SCALES = (0.25, 0.5, 1.0, 2.0)
TOL, CAP = 1e-8, 300
if STAGE == "small":
    S, SIZES, SPREADS, BASE, WHICH = 0, (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20), \
        (0.5, 1.0, 2.0, 3.0), "normal", CLASSES
    TARGETS = ("round trip", "dirichlet 1", "dirichlet 0.3")
elif STAGE == "bases":
    S, SIZES, SPREADS, BASE, WHICH = 1, (2, 3, 4, 5, 8, 12, 20), \
        (0.5, 1.0, 2.0), "gumbel", CLASSES[:2]
    TARGETS = ("round trip", "dirichlet 1")
else:
    S, SIZES, SPREADS, BASE, WHICH = 2, (50, 100, 200, 1000, 10000, 100000), \
        (1.0,), "normal", CLASSES
    TARGETS = ("round trip", "dirichlet 1")


def field(g, n, j, k, r):
    rng = np.random.default_rng([2028, S, g, n, j, k, r])
    mu = rng.normal(size=n) * SPREADS[k]
    mu -= mu.mean()
    D = 0.5 + rng.random(n)
    scale = SCALES[j]
    free = {"dirichlet 1": rng.dirichlet(np.ones(n)),
            "dirichlet 0.3": rng.dirichlet(np.full(n, 0.3))}
    name = CLASSES[g]
    extra = {} if BASE == "normal" else {"base": BASE}
    if name == "independent":
        return mu, free, dict(D=D, **extra)
    if name == "factor rank 1":
        return mu, free, dict(V=rng.normal(size=(n, 1)) * scale, D=D, **extra)
    if name == "factor rank 2":
        return mu, free, dict(V=rng.normal(size=(n, 2)) * scale, D=D, **extra)
    cluster = np.arange(n) // (2 if n < 50 else 10)
    n_c = int(cluster.max()) + 1
    loading = (0.5 + rng.random(n)) * scale
    if name == "blocks":
        return mu, free, dict(structure=Blocks(cluster=cluster,
                                               loading=loading, D=D))
    if name == "tree":
        g1 = (n_c + 1) // 2
        parent = np.concatenate([n_c + np.arange(n_c) // 2,
                                 np.full(g1, n_c + g1), [-1]])
        strength = np.concatenate([np.zeros(n_c),
                                   (0.2 + 0.4 * rng.random(g1 + 1)) * scale])
        return mu, free, dict(structure=Tree(cluster=cluster, loading=loading,
                                             D=D, parent=parent,
                                             strength=strength))
    coupling = 0.5 * rng.random(n) * scale
    return mu, free, dict(structure=Nested(cluster=cluster, loading=loading,
                                           D=D, coupling=coupling, gamma=0.8))


def price(mu, kw):
    if "structure" in kw:
        return dispatch_probabilities(mu, kw["structure"])
    return race_probabilities(mu, **kw)


def solve(target, kw, rule, cap=CAP):
    """Copy of the shipped loop for the field's class under one step
    rule. Returns (mu, sweeps, last residual); sweeps is cap + 1 when
    the residual is still above TOL after cap sweeps."""
    n = len(target)
    structured = "structure" in kw
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    if structured:
        totvar = structure_variances(kw["structure"])
    if rule == "sign":
        alpha = np.ones(n)
    elif structured:
        alpha = 0.7
    else:
        alpha = 1.0 if n > 2 else 0.7
    last, prev, err = np.inf, None, np.inf
    for it in range(cap):
        if structured:
            phat = dispatch_probabilities(mu, kw["structure"])
        else:
            phat, sl = race_probabilities(mu, return_slopes=True, **kw)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        err = float(np.abs(resid).max())
        if err < TOL:
            return mu, it + 1, err
        if rule == "sign":
            if prev is not None:
                flip = resid * prev < 0
                alpha = np.where(flip, np.maximum(alpha * 0.5, 0.05),
                                 np.minimum(alpha * 1.5, 1.0))
            prev = resid
        elif structured:
            if err > last:
                alpha = max(alpha * 0.5, 0.05)
            last = err
        if structured:
            phat, sl = race_probabilities(mu, D=totvar, return_slopes=True)
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return mu, cap + 1, err


def work(task):
    g, n, j, k, r = task
    warnings.simplefilter("ignore")
    try:
        os.nice(0)
    except OSError:
        pass
    out = []
    mu, free, kw = field(g, n, j, k, r)
    try:
        p = price(mu, kw)
    except Exception as e:                      # noqa: BLE001
        return [dict(cls=CLASSES[g], n=n, scale=SCALES[j], spread=SPREADS[k],
                     rep=r, target="forward", error=repr(e)[:80])]
    for kind in TARGETS:
        target = p if kind == "round trip" else free[kind]
        target = target / target.sum()
        row = dict(cls=CLASSES[g], n=n, scale=SCALES[j], spread=SPREADS[k],
                   rep=r, target=kind, min_target=float(target.min()), error="")
        if not np.all(target > 0):
            row["error"] = "zero target"
            out.append(row)
            continue
        for rule in ("shipped", "sign"):
            t0 = time.time()
            try:
                m, its, err = solve(target, kw, rule)
            except Exception as e:              # noqa: BLE001
                row["error"] = f"{rule}: {repr(e)[:60]}"
                m, its, err = mu * np.nan, CAP + 1, np.nan
            row[f"{rule}_sweeps"] = its
            row[f"{rule}_resid"] = err
            row[f"{rule}_seconds"] = round(time.time() - t0, 3)
            row[f"{rule}_mu_err"] = (float(np.abs(m - mu).max())
                                     if kind == "round trip" else "")
        out.append(row)
    return out


FIELDS = ["cls", "n", "scale", "spread", "rep", "target", "min_target",
          "shipped_sweeps", "shipped_resid", "shipped_seconds", "shipped_mu_err",
          "sign_sweeps", "sign_resid", "sign_seconds", "sign_mu_err", "error"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(here, f"sign_rule_{STAGE}.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    print(f"stage {STAGE}: base {BASE}, {SEEDS} replicates, spreads {SPREADS}, "
          f"targets {TARGETS}, {CAP} sweeps allowed, {WORKERS} workers")
    print(f"{'class':14s} {'n':>6s} {'scale':>5s} {'runs':>5s} | "
          f"shipped: fail stuck  med/max | sign: fail stuck  med/max  "
          f"worst resid  worst |mu err| | sign slower  seconds")
    bad, tot = [], dict(runs=0, sh=0, sg=0, sh_stuck=0, sg_stuck=0, err=0)
    with mp.Pool(WORKERS) as pool:
        for name in WHICH:
            g = CLASSES.index(name)
            cap0 = 60 if g < 3 else 120
            for n in SIZES:
                for j, scale in enumerate(SCALES):
                    if name == "independent" and j > 0:
                        continue
                    t0 = time.time()
                    tasks = [(g, n, j, k, r) for k in range(len(SPREADS))
                             for r in range(SEEDS)]
                    rows = [x for res in pool.imap(work, tasks) for x in res]
                    for x in rows:
                        w.writerow(x)
                    f.flush()
                    errs = [x for x in rows if x["error"]]
                    ok = [x for x in rows if not x["error"]]
                    tot["err"] += len(errs)
                    for x in errs:
                        bad.append(f"  ERROR {x['cls']} n={x['n']} scale={x['scale']} "
                                   f"spread={x['spread']} rep={x['rep']} "
                                   f"{x['target']}: {x['error']}")
                    if not ok:
                        continue
                    sh = np.array([x["shipped_sweeps"] for x in ok])
                    sg = np.array([x["sign_sweeps"] for x in ok])
                    slower = int(np.sum(sg > sh + 10))
                    mer = [x["sign_mu_err"] for x in ok
                           if x["sign_mu_err"] != "" and x["sign_sweeps"] <= CAP]
                    for x in ok:
                        if x["sign_sweeps"] > cap0 or x["sign_sweeps"] > x["shipped_sweeps"] + 10:
                            bad.append(
                                f"  {x['cls']} n={x['n']} scale={x['scale']} "
                                f"spread={x['spread']} rep={x['rep']} {x['target']}: "
                                f"sign {x['sign_sweeps']} sweeps (resid "
                                f"{x['sign_resid']:.1e}), shipped "
                                f"{x['shipped_sweeps']} (resid {x['shipped_resid']:.1e}), "
                                f"min target {x['min_target']:.1e}")
                    tot["runs"] += len(ok)
                    tot["sh"] += int(np.sum(sh > cap0))
                    tot["sg"] += int(np.sum(sg > cap0))
                    tot["sh_stuck"] += int(np.sum(sh > CAP))
                    tot["sg_stuck"] += int(np.sum(sg > CAP))
                    print(f"{name:14s} {n:6d} "
                          f"{('-' if name == 'independent' else f'{scale:g}'):>5s} "
                          f"{len(ok):5d} | {int(np.sum(sh > cap0)):13d} "
                          f"{int(np.sum(sh > CAP)):5d}  {int(np.median(sh)):3d}/{sh.max():<3d} | "
                          f"{int(np.sum(sg > cap0)):10d} {int(np.sum(sg > CAP)):5d}  "
                          f"{int(np.median(sg)):3d}/{sg.max():<3d}  "
                          f"{max(x['sign_resid'] for x in ok):.1e}      "
                          f"{(max(mer) if mer else float('nan')):.1e}   | "
                          f"{slower:11d}  {time.time() - t0:7.0f}", flush=True)
    print(f"\nsign rule fails, or needs more than ten sweeps beyond shipped: {len(bad)}")
    for line in bad:
        print(line)
    print(f"\ntotals: {tot['runs']} runs; above the shipped cap: shipped {tot['sh']}, "
          f"sign {tot['sg']}; above {CAP} sweeps: shipped {tot['sh_stuck']}, "
          f"sign {tot['sg_stuck']}; errors {tot['err']}")
