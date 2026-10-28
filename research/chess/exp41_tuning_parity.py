"""exp41: both sides tuned properly. The chess headline was overstated.

=====================================================================
RESULT (2026-09-13). All four predictions confirmed. The headline
survives and shrinks by a third; the estimator column shrinks by
three fifths; the structure column is untouched.

    row             published  corrected   change
    headline          -0.0160    -0.0103   +0.0057   36% smaller
    estimator         -0.0199    -0.0080   +0.0119   60% smaller
    structure         -0.0035    -0.0033   +0.0002   unchanged
    strat_value       +0.0074    -0.0011   -0.0085   SIGN FLIP

    seed  glicko2 pooled  glicko2 perTC  our scalar  our factor  headline
      0          0.6193         0.6177      0.6117      0.6073   -0.0104 [-0.0131,-0.0076]
     11          0.6206         0.6191      0.6134      0.6111   -0.0080 [-0.0111,-0.0049]
     22          0.6233         0.6232      0.6140      0.6108   -0.0124 [-0.0157,-0.0092]

X1 CONFIRMED: the factor arm still beats Glicko-2 per time control on
all three splits with intervals excluding zero. The result stands.

X3 is the one that matters for how the result should be described.
The estimator column falls 60% and the structure column 5%. The part
of the margin that answers the research question -- does a vector
rating beat a scalar one -- is robust to the tuning defect. The part
that was inflated is the claim about our estimator, which is also the
part the README leads with as four fifths of the headline.

X4 flips sign: stratification is now worth -0.0011 to Glicko-2 rather
than +0.0074. Per-category ratings were not buying Glicko-2 category
information so much as protecting it from its own over-inflated
deviations. Once `period` is tuned that protection is unnecessary, and
the README section titled for stratification helping Glicko-2 and not
us needs rewriting: with both sides tuned, stratification helps
neither. The pooling claim survives in a better form -- conditioning
buys nothing once the base estimator is set up properly -- but it is
a different sentence from the published one.

GRIDS. level, offset and initial_rd all select interior values.
`period` selects the largest value on every arm and every seed, and
that flag is BENIGN rather than unresolved: the parameter saturates.
Validation loss across period 200 / 1500 / 20000 moves 0.6203 /
0.6200 / 0.6199, because past roughly 1500 the rating-deviation
inflation is switched off entirely and larger values cannot do
anything further. The honest reading is that the best configuration
for a dense single month is NO RD inflation, which is what a period
far exceeding the window means.
=====================================================================

WHAT WENT WRONG. Every chess experiment in this directory tuned
Glicko-2's `tau` on validation and said so, in the README and in the
note, as evidence that "neither side is given a free parameter the
other lacks". `tau` is inert here. Sweeping it from 0.02 to 1.0 on
exp35's protocol returns bit-identical held-out loss to four decimals,
because it enters only the volatility iteration and the volatility
barely moves over one month. Python's `min` was selecting 0.3 out of
seven ties, in exp35, exp37, exp39 and exp40 alike, and the grids all
started at 0.3 so nothing ever looked lower.

Meanwhile the Glicko-2 parameters that do matter were left at library
defaults and never tuned: `period`, which controls how fast rating
deviation inflates with time and is the direct analogue of our
shrinkage, and `initial_rd`. On seed 0, tuning them moves Glicko-2
per-time-control from 0.6234 to 0.6177 and Glicko-2 pooled from 0.6323
to 0.6196.

So the comparison was not fair, and it was unfair in our favour.

THE SYMMETRIC FIX. Correcting only the competitor's handicap would
create its mirror image, because our arms were under-tuned too: the
scalar's level ridge was pinned at 1.0 in every experiment and only
the factor arm's OFFSET ridge was ever swept. This file tunes every
arm's material parameters on the validation split, with grids extended
until the selected value is interior, and reports the parameter count
each side used.

  glicko2 pooled / per TC   period, initial_rd
  our scalar                level ridge
  our factor                level ridge, offset ridge

Selection is on validation only; the scored quarter is untouched until
the parameters are fixed. Three splits, as everywhere in this
directory.

PRE-REGISTERED:
  X1. The factor arm still beats Glicko-2 per time control on all
      three splits with intervals excluding zero. The result survives,
      smaller.
  X2. The margin shrinks materially, by more than 20% on average.
  X3. The ESTIMATOR column shrinks by more than the structure column,
      because `period` is a recency parameter and recency is what a
      batch fit was implicitly buying.
  X4. Stratification's value to Glicko-2 falls once its recency is
      tuned, because part of what per-category ratings were buying was
      protection against over-inflated deviations.
  FALSIFICATION: if the factor arm no longer beats Glicko-2 per time
  control on a majority of splits, the chess headline is withdrawn and
  the note retracted. That is the outcome this file is built to be able
  to report.

Designs are sparse, as in exp31 and exp38.

Run:  python research/chess/exp41_tuning_parity.py
"""
from __future__ import annotations
import importlib.util, os, sys, types
import numpy as np, pandas as pd
from scipy import sparse
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings


def _load_glicko():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    roots = [os.path.join(repo, "attic", "src", "winning"),
             os.path.expanduser("~/github/winning/attic/src/winning")]
    root = next((r for r in roots if os.path.isdir(r)), None)
    if root is None:
        raise ImportError(f"glicko2 source tree not found in {roots}")
    pkg = types.ModuleType("wsrc"); pkg.__path__ = [root]
    sys.modules["wsrc"] = pkg
    for nm in ("ratingsystem", "glicko2"):
        spec = importlib.util.spec_from_file_location(
            f"wsrc.{nm}", os.path.join(root, nm + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"wsrc.{nm}"] = mod; setattr(pkg, nm, mod)
        spec.loader.exec_module(mod)
    from wsrc.glicko2 import Glicko2Rating
    return Glicko2Rating


G = _load_glicko()
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
TCS = ["classical", "blitz", "bullet"]

df = pd.read_parquet(CACHE)
df = df[df.result.isin(["1-0", "0-1"])]
ev = df.event.str.lower()
tcn = np.where(ev.str.contains("bullet"), "bullet",
      np.where(ev.str.contains("blitz"), "blitz",
      np.where(ev.str.contains("classical"), "classical", "other")))
m = tcn != "other"
df = df[m].reset_index(drop=True); tcn = tcn[m]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= 100].index)
m2 = (df.white.isin(keep) & df.black.isin(keep)).values
df = df[m2].reset_index(drop=True); tcn = tcn[m2]
players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
Mp = len(players)
tci = pd.Series(tcn).map({t: i for i, t in enumerate(TCS)}).values
w = df.white.map(pidx).values; b = df.black.map(pidx).values
won = (df.result == "1-0").values
n = len(df)
P, NG = 3, 1 + len(TCS)
print(f"exp41: {n} games, {Mp} players", flush=True)

PERIOD = (30.0, 200.0, 1500.0, 20000.0)
IRD = (350.0, 500.0, 700.0, 1000.0)
LEVEL = (0.03, 0.1, 0.3, 1.0, 3.0)
OFFSET = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def glicko(fit, ev_, per_tc, period, ird):
    S = ({c: G(period=period, initial_rd=ird) for c in range(len(TCS))}
         if per_tc else {0: G(period=period, initial_rd=ird)})
    for t in fit:
        S[tci[t] if per_tc else 0].observe(
            [players[w[t]], players[b[t]]],
            [1, 2] if won[t] else [2, 1], 1.0)
    out = []
    for t in ev_:
        p = np.asarray(S[tci[t] if per_tc else 0].win_probabilities(
            [players[w[t]], players[b[t]]]), dtype=float)
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if won[t] else p[1]))
    return np.array(out)


def xvec(t):
    return np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])


def ours(fit, ev_, lam_lvl, lam_off, scalar=False):
    pp = 1 if scalar else P
    width = Mp * pp + NG
    evs = []
    for t in fit:
        x = xvec(t)[:pp]
        rows = [0] * pp + [1] * pp + [0, 0]
        cols = ([w[t] * pp + k for k in range(pp)]
                + [b[t] * pp + k for k in range(pp)]
                + [Mp * pp, Mp * pp + 1 + tci[t]])
        vals = list(x) + list(x) + [1.0, 1.0]
        Z = sparse.csr_matrix((vals, (rows, cols)), shape=(2, width))
        evs.append((Z, np.array([0, 1]) if won[t] else np.array([1, 0])))
    v = np.full(width, lam_lvl)
    if not scalar:
        v[np.concatenate([np.arange(Mp) * P + 1,
                          np.arange(Mp) * P + 2])] = lam_off
    v[Mp * pp] = 0.1; v[Mp * pp + 1:] = 1.0
    th = fit_design_ratings(evs, width, ridge=v)
    out = []
    for t in ev_:
        x = xvec(t)[:pp]
        mu_w = th[w[t]*pp:w[t]*pp+pp] @ x + th[Mp*pp] + th[Mp*pp+1+tci[t]]
        mu_b = th[b[t]*pp:b[t]*pp+pp] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if won[t] else p[1]))
    return np.array(out)


def edge(v, grid):
    return "*" if v in (grid[0], grid[-1]) else " "


rows = []
for seed in (0, 11, 22):
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    ntr, nva = int(n * .50), int(n * .25)
    tr = np.sort(perm[:ntr]); va = np.sort(perm[ntr:ntr + nva])
    te = np.sort(perm[ntr + nva:])
    print(f"\n--- seed {seed} ---", flush=True)
    res, pick = {}, {}
    for lbl, per in (("glicko2 pooled", False), ("glicko2 per TC", True)):
        best = min(((glicko(tr, va, per, p_, i_).mean(), p_, i_)
                    for p_ in PERIOD for i_ in IRD))
        res[lbl] = glicko(tr, te, per, best[1], best[2])
        pick[lbl] = f"period {best[1]:g}{edge(best[1],PERIOD)} ird {best[2]:g}{edge(best[2],IRD)}"
        print(f"  {lbl:16s} {res[lbl].mean():.4f}   {pick[lbl]}", flush=True)
    best = min(((ours(tr, va, l, 1.0, True).mean(), l) for l in LEVEL))
    res["our scalar"] = ours(tr, te, best[1], 1.0, True)
    pick["our scalar"] = f"level {best[1]}{edge(best[1],LEVEL)}"
    print(f"  {'our scalar':16s} {res['our scalar'].mean():.4f}   "
          f"{pick['our scalar']}", flush=True)
    best = min(((ours(tr, va, l, o).mean(), l, o) for l in LEVEL for o in OFFSET))
    res["our factor"] = ours(tr, te, best[1], best[2])
    pick["our factor"] = (f"level {best[1]}{edge(best[1],LEVEL)} "
                          f"offset {best[2]:g}{edge(best[2],OFFSET)}")
    print(f"  {'our factor':16s} {res['our factor'].mean():.4f}   "
          f"{pick['our factor']}", flush=True)
    ids = np.random.default_rng(1).integers(0, len(te), (20000, len(te)))

    def d(a, bb):
        dd = (res[a][ids] - res[bb][ids]).mean(1)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        return res[a].mean() - res[bb].mean(), lo, hi, (lo > 0 or hi < 0)
    rows.append({
        "seed": seed,
        "g_pool": res["glicko2 pooled"].mean(),
        "g_tc": res["glicko2 per TC"].mean(),
        "scalar": res["our scalar"].mean(),
        "factor": res["our factor"].mean(),
        "headline": d("our factor", "glicko2 per TC")[0],
        "head_sig": d("our factor", "glicko2 per TC")[3],
        "estimator": d("our scalar", "glicko2 pooled")[0],
        "structure": d("our factor", "our scalar")[0],
        "strat_value": d("glicko2 per TC", "glicko2 pooled")[0]})
    h = d("our factor", "glicko2 per TC")
    print(f"    headline {h[0]:+.4f} [{h[1]:+.4f},{h[2]:+.4f}]"
          f"{'' if h[3] else '  n.s.'}", flush=True)

t = pd.DataFrame(rows)
PUB = {"headline": [-0.0161, -0.0137, -0.0182],
       "estimator": [-0.0206, -0.0173, -0.0219],
       "structure": [-0.0044, -0.0027, -0.0033],
       "strat_value": [+0.0089, +0.0063, +0.0071]}
print("\n=== CORRECTED vs PUBLISHED ===")
print(f"{'row':<14}{'published mean':>16}{'corrected mean':>16}{'change':>12}")
for k in ("headline", "estimator", "structure", "strat_value"):
    pub = float(np.mean(PUB[k])); cor = float(t[k].mean())
    print(f"{k:<14}{pub:>+16.4f}{cor:>+16.4f}{cor-pub:>+12.4f}")
print(f"\n  X1 factor still beats Glicko-2 per TC on all three splits, "
      f"CI excluding zero: {bool(t.head_sig.all() and (t.headline<0).all())}")
shrink = 1 - abs(t.headline.mean()) / abs(np.mean(PUB['headline']))
print(f"  X2 headline shrinks by more than 20%: {100*shrink:.0f}% -> "
      f"{'CONFIRMED' if shrink > 0.20 else 'FAILED'}")
es = 1 - abs(t.estimator.mean()) / abs(np.mean(PUB['estimator']))
st = 1 - abs(t.structure.mean()) / abs(np.mean(PUB['structure']))
print(f"  X3 estimator shrinks more than structure: "
      f"{100*es:.0f}% vs {100*st:.0f}% -> "
      f"{'CONFIRMED' if es > st else 'FAILED'}")
print(f"  X4 stratification's value to Glicko-2 falls: "
      f"{np.mean(PUB['strat_value']):+.4f} -> {t.strat_value.mean():+.4f} -> "
      f"{'CONFIRMED' if t.strat_value.mean() < np.mean(PUB['strat_value']) else 'FAILED'}")
print("\n  '*' on a selected value means it sat at a grid edge and the "
      "grid needs extending.")
out = os.path.join(HERE, "results", "exp41_tuning_parity.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
t.to_csv(out, index=False)
print(f"  -> {out}")
