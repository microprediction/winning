"""exp40: online against online, which is the comparison the chess
headline has been avoiding.

The chess result is that a partially pooled factor rating beats
Lichess's deployed per-time-control Glicko-2 by 0.0137 to 0.0182 nats,
and that roughly four fifths of the margin is the ESTIMATOR rather
than the factor structure. The README has ranked one objection first
since the result was published: our estimator is a batch MAP fit that
sees the training games at once and can revisit them, while Glicko-2
processes each game once because a production system must. Some
unknown share of that estimator column is the batch privilege rather
than a better estimator.

This measures the share. Both sides run ONLINE, one pass, no revisits.

PROTOCOL. Prequential, which is what a deployed rater actually faces:
walk the month in order, and for every game predict it from the state
built out of strictly earlier games, then learn from it. Every game
after the warmup contributes to the loss, and no game informs its own
prediction.

  warmup   first 25%   consumed, not scored
  tune     next 25%    scored prequentially to pick one hyperparameter
  report   final 50%   scored prequentially at the tuned value

Chronology comes from row order in the month's PGN. The header table
carries no timestamp, so this is the available ordering and it is the
one the file was written in.

HYPERPARAMETER PARITY, corrected after the first run. The first
version tuned Glicko-2's `tau`, which exp41 then showed is INERT here
(0.02 to 1.0 give bit-identical loss), so Glicko-2 effectively ran at
library defaults while our tracker got a real knob. That is the same
defect exp41 found in the batch experiments, reproduced in a new
place, and the first run's numbers are void.

Both sides now get TWO tuned parameters covering the same two
concepts, recency and initial uncertainty:

    Glicko-2          period, initial_rd
    AbilityTracker    drift,  init_var

Grids are extended until the selected value is interior; a selection
at a grid end is flagged and the run should not be quoted until it is
cleared.

WHAT THIS CANNOT TEST, stated because it bounds the conclusion.
AbilityTracker takes ids and groups but no per-entrant covariates, so
it cannot express the partially pooled factor structure (one level
plus shrunk offsets). The online arms available are POOLED (one rating
per player) and STRATIFIED (one rating per player per time control,
which is Lichess's architecture). This experiment therefore measures
the ESTIMATOR column online and says nothing about the structure
column, which remains a batch-only result.

ARMS, all online, all one pass:
  glicko2 pooled      the weak baseline
  glicko2 per TC      Lichess's deployed architecture
  tracker pooled      our estimator, pooled
  tracker per TC      our estimator, stratified

PRE-REGISTERED PREDICTIONS:
  W1. tracker pooled beats glicko2 pooled online, CI excluding zero.
      The estimator is better on its own merits and not only because
      it was allowed to revisit.
  W2. The online margin is SMALLER than the batch estimator column.
      That column is -0.0080 after exp41's tuning correction, NOT the
      -0.0206 originally published, and the corrected figure is what
      this is measured against. The difference is the batch privilege.
  W3. Stratification helps Glicko-2 more than it helps our tracker.
      The batch pattern to reproduce is exp41's CORRECTED one, where
      stratification is worth -0.0011 to a properly tuned Glicko-2
      rather than the +0.0089 originally published. This tests whether the pooling story is
      about the estimator's quality or about batch fitting.
  FALSIFICATION, and it is severe: if tracker pooled does NOT beat
  glicko2 pooled online, then the estimator column is entirely the
  batch privilege. The headline would have to be restated as "a batch
  fit beats an online one", which is a much weaker claim and arguably
  not a rating-systems result at all. I would then withdraw the
  estimator framing from the chess note and the README.

Absolute numbers here are NOT comparable with exp35's. That used a
random 50/25/25 split and scored a held-out quarter from a frozen
state; this is prequential over a chronological month. The comparison
that carries across is between arms within this file.

Run:  python research/chess/exp40_online_vs_batch.py
"""
from __future__ import annotations
import importlib.util, os, sys, types
import numpy as np, pandas as pd
from winning.ratings import AbilityTracker
from winning.ratings import select


def _load_glicko():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    roots = [os.path.join(repo, "src", "winning"),
             os.path.expanduser("~/github/winning/src/winning")]
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


Glicko2Rating = _load_glicko()
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
TCS = ["classical", "blitz", "bullet"]

df = pd.read_parquet(CACHE)
df = df[df.result.isin(["1-0", "0-1"])]
ev = df.event.str.lower()
tcname = np.where(ev.str.contains("bullet"), "bullet",
         np.where(ev.str.contains("blitz"), "blitz",
         np.where(ev.str.contains("classical"), "classical", "other")))
m = tcname != "other"
df = df[m].reset_index(drop=True); tcname = tcname[m]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= 100].index)
m2 = (df.white.isin(keep) & df.black.isin(keep)).values
df = df[m2].reset_index(drop=True); tcname = tcname[m2]
tci = pd.Series(tcname).map({t: i for i, t in enumerate(TCS)}).values
W = df.white.values; B = df.black.values
white_won = (df.result == "1-0").values
n = len(df)
WARM, TUNE = int(n * 0.25), int(n * 0.50)
print(f"exp40: {n} games in chronological order, {len(keep)} players")
print(f"  warmup 0-{WARM}, tune {WARM}-{TUNE}, report {TUNE}-{n}", flush=True)


def prequential_glicko(params, per_tc, lo, hi):
    """One pass. Predict from strictly earlier games, then learn."""
    period, ird = params
    S = ({c: Glicko2Rating(period=period, initial_rd=ird)
          for c in range(len(TCS))} if per_tc
         else {0: Glicko2Rating(period=period, initial_rd=ird)})
    out = []
    for t in range(hi):
        s = S[tci[t] if per_tc else 0]
        names = [W[t], B[t]]
        if t >= lo:
            p = np.asarray(s.win_probabilities(names), dtype=float)
            p = np.clip(p, 1e-12, None); p /= p.sum()
            out.append(-np.log(p[0] if white_won[t] else p[1]))
        s.observe(names, [1, 2] if white_won[t] else [2, 1], 1.0)
    return np.array(out)


def prequential_tracker(params, per_tc, lo, hi):
    drift, ivar = params
    S = ({c: AbilityTracker(drift=drift, init_var=ivar)
          for c in range(len(TCS))} if per_tc
         else {0: AbilityTracker(drift=drift, init_var=ivar)})
    out = []
    for t in range(hi):
        s = S[tci[t] if per_tc else 0]
        names = [W[t], B[t]]
        if t >= lo:
            p = np.asarray(s.predict(names, t=float(t)), dtype=float)
            p = np.clip(p, 1e-12, None); p /= p.sum()
            out.append(-np.log(p[0] if white_won[t] else p[1]))
        s.observe(names, t=float(t), winner=0 if white_won[t] else 1)
    return np.array(out)


GL_GRID = [(p, i) for p in (30.0, 200.0, 1500.0, 20000.0)
                   for i in (350.0, 500.0, 700.0, 1000.0)]
TR_GRID = [(d, v) for d in (0.0, 0.0002, 0.002, 0.01, 0.04, 0.15)
                  for v in (1.0, 4.0, 9.0, 16.0)]   # two knobs each side
res, chosen = {}, {}
for lbl, fn, grid, per in (
        ("glicko2 pooled", prequential_glicko, GL_GRID, False),
        ("glicko2 per TC", prequential_glicko, GL_GRID, True),
        ("tracker pooled", prequential_tracker, TR_GRID, False),
        ("tracker per TC", prequential_tracker, TR_GRID, True)):
    tuned = {g: fn(g, per, WARM, TUNE).mean() for g in grid}
    best = min(tuned.items(), key=lambda kv: kv[1])[0]
    res[lbl] = fn(best, per, TUNE, n)
    chosen[lbl] = best
    # Per-AXIS edge reporting. A tuple grid hides an axis sitting at its
    # end, which is the weakness exp41's tau defect teaches us not to
    # tolerate: profile each axis by its best achievable score.
    notes = []
    for ax in (0, 1):
        vals = sorted({g[ax] for g in grid})
        prof = {v: min(sc for g, sc in tuned.items() if g[ax] == v)
                for v in vals}
        _, rep = select(prof, f"axis{ax}", order=vals)
        notes.append(str(rep))
    print(f"  {lbl:16s} REPORT {res[lbl].mean():.4f}   "
          f"(param {best}, tune {tuned[best]:.4f})", flush=True)
    for nt in notes:
        print(f"      {nt}", flush=True)

k = min(len(v) for v in res.values())
for key in res:
    res[key] = res[key][:k]
ids = np.random.default_rng(1).integers(0, k, (20000, k))


def cmp(a, b):
    d = (res[a][ids] - res[b][ids]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = (lo > 0 or hi < 0)
    print(f"  {a:16s} vs {b:16s} {res[a].mean()-res[b].mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  P {np.mean(d<0):.3f}"
          f"{'' if sig else '  n.s.'}")
    return res[a].mean() - res[b].mean(), sig


print(f"\n  --- W1: is our estimator better ONLINE? ({k} scored games) ---")
d1, s1 = cmp("tracker pooled", "glicko2 pooled")
print("  --- W3: what does stratification buy each side? ---")
dg, _ = cmp("glicko2 per TC", "glicko2 pooled")
dt, _ = cmp("tracker per TC", "tracker pooled")
print("  --- against the deployed architecture ---")
d2, s2 = cmp("tracker pooled", "glicko2 per TC")

BATCH_EST = -0.0080   # exp41 corrected; was -0.0206 when mis-tuned
print(f"\n  W1 tracker beats glicko2 pooled online: {d1:+.4f}"
      f"{'' if s1 else ' (n.s.)'} -> "
      f"{'CONFIRMED' if (d1 < 0 and s1) else 'FAILED'}")
print(f"  W2 online margin smaller than batch {BATCH_EST:+.4f}: "
      f"{d1:+.4f} -> {'CONFIRMED' if d1 > BATCH_EST else 'FAILED'}")
if d1 < 0:
    print(f"     batch privilege = {BATCH_EST:+.4f} - {d1:+.4f} = "
          f"{BATCH_EST-d1:+.4f} nats, "
          f"{100*(BATCH_EST-d1)/BATCH_EST:.0f}% of the batch estimator column")
print(f"  W3 stratification helps Glicko-2 more than us: "
      f"{dg:+.4f} vs {dt:+.4f} -> "
      f"{'CONFIRMED' if dg < dt else 'FAILED'}")
print("\n  NOTE: absolute numbers are not comparable with exp35's batch"
      "\n  figures -- different protocol (prequential chronological vs a"
      "\n  random split scored from a frozen state). Compare arms within"
      "\n  this file.")
out = os.path.join(HERE, "results", "exp40_online.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(res).to_csv(out, index=False)
print(f"  per-game losses -> {out}")
