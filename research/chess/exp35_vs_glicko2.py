"""exp35: beat the REAL Lichess architecture, not my straw version.

exp30 showed the factor rating beating a stratified-by-time-control
arm — but that arm was MY estimator run separately per control, not
Glicko-2, which is what Lichess actually runs. A referee would rightly
say we beat our own stratification. This fields the real thing.

ARMS, all under one protocol (fit on the training half in
chronological order, freeze, predict the held-out half):

  glicko2 pooled     one Glicko-2 rating per player, all controls
                     mixed — the "what if Lichess pooled" counterfactual
  glicko2 per TC     a separate Glicko-2 rating per time control —
                     LICHESS'S ACTUAL ARCHITECTURE
  factor             ours: level + bullet/blitz offsets, per-feature
                     ridge, shared colour terms

FAIRNESS. Glicko-2's tau is tuned on a validation slice exactly as our
ridge is; neither system gets a hand-picked knob. Every arm predicts
the same games from the same information. Glicko-2 supplies its own
win probabilities through the winning package's implementation, so the
comparison is architecture vs architecture, not my probit vs their
logistic.

PRE-REGISTERED PREDICTIONS:
  P1. factor beats glicko2-per-TC (Lichess's architecture) with a CI
      excluding zero, on every split.
  P2. glicko2-pooled is NOT significantly worse than glicko2-per-TC —
      replicating exp30's finding that stratification buys nothing,
      now against the real implementation rather than my stand-in.
  P3. Glicko-2 pooled and our scalar land close; the factor margin
      over Glicko-2-per-TC is therefore mostly the pooling mechanism,
      not the estimator.
  FALSIFICATION: if glicko2-per-TC beats the factor arm, the "beats
  Lichess" claim is withdrawn and exp30's stratified arm is recorded
  as having been too weak a competitor.

Run:  python experiments/exp35_vs_glicko2.py [split_seed]
"""
from __future__ import annotations
import importlib.util, os, sys, types
import numpy as np, pandas as pd
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map

def _load_glicko():
    root = os.path.expanduser("~/github/winning/src/winning")
    if not os.path.isdir(root):
        root = os.path.expanduser("~/github/winning/winning")
    pkg = types.ModuleType("wsrc"); pkg.__path__ = [root]
    sys.modules["wsrc"] = pkg
    for n in ("ratingsystem", "glicko2"):
        p = os.path.join(root, n + ".py")
        spec = importlib.util.spec_from_file_location(f"wsrc.{n}", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"wsrc.{n}"] = m; setattr(pkg, n, m)
        spec.loader.exec_module(m)
    from winning.glicko2 import Glicko2Rating
    return Glicko2Rating

Glicko2Rating = _load_glicko()
CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
TCS = ['classical', 'blitz', 'bullet']
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

df = pd.read_parquet(CACHE)
df = df[df.result.isin(['1-0', '0-1'])]
ev = df.event.str.lower()
tcname = np.where(ev.str.contains('bullet'), 'bullet',
         np.where(ev.str.contains('blitz'), 'blitz',
         np.where(ev.str.contains('classical'), 'classical', 'other')))
m = tcname != 'other'
df = df[m].reset_index(drop=True); tcname = tcname[m]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= 100].index)
m2 = (df.white.isin(keep) & df.black.isin(keep)).values
df = df[m2].reset_index(drop=True); tcname = tcname[m2]
players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
Mp = len(players)
tci = pd.Series(tcname).map({t: i for i, t in enumerate(TCS)}).values
w = df.white.map(pidx).values; b = df.black.map(pidx).values
white_won = (df.result == '1-0').values
n = len(df)
rng = np.random.default_rng(SEED); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = np.sort(perm[:ntr]), np.sort(perm[ntr:ntr+nva]), np.sort(perm[ntr+nva:])
print(f"split seed {SEED}: {n} games, {Mp} players; "
      f"train {len(tr)} val {len(va)} TEST {len(te)}")

def run_glicko(idx_fit, idx_eval, tau, per_tc):
    """Chronological online fit, then frozen prediction."""
    systems = ({c: Glicko2Rating(tau=tau) for c in range(len(TCS))}
               if per_tc else {0: Glicko2Rating(tau=tau)})
    for t in idx_fit:
        s = systems[tci[t] if per_tc else 0]
        names = [players[w[t]], players[b[t]]]
        ranks = [1, 2] if white_won[t] else [2, 1]
        s.observe(names, ranks, 1.0)
    out = []
    for t in idx_eval:
        s = systems[tci[t] if per_tc else 0]
        p = np.asarray(s.win_probabilities(
            [players[w[t]], players[b[t]]]), dtype=float)
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

def xvec(t):
    return np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])

def run_factor(idx_fit, idx_eval, lam_off, scalar=False):
    """scalar=True gives our estimator WITHOUT the offsets -- the arm
    that separates 'our estimator is better than Glicko-2' from 'the
    factor structure helps'. Without it the headline margin conflates
    the two."""
    P, NG = (1 if scalar else 3), 1 + len(TCS)
    width = Mp*P + NG
    ev_ = []
    for t in idx_fit:
        x = xvec(t)[:P]; Z = np.zeros((2, width))
        Z[0, w[t]*P:w[t]*P+P] = x; Z[1, b[t]*P:b[t]*P+P] = x
        Z[0, Mp*P] = 1.0; Z[0, Mp*P+1+tci[t]] = 1.0
        ev_.append((Z, np.array([0,1]) if white_won[t] else np.array([1,0])))
    v = np.full(width, 1.0)
    if not scalar:
        v[np.concatenate([np.arange(Mp)*P+1, np.arange(Mp)*P+2])] = lam_off
    v[Mp*P] = 0.1; v[Mp*P+1:] = 1.0
    th = linear_ability_map(ev_, width, ridge=v)
    out = []
    for t in idx_eval:
        x = xvec(t)[:P]
        mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+tci[t]]
        mu_b = th[b[t]*P:b[t]*P+P] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

res = {}
for label, per_tc in (("glicko2 pooled", False), ("glicko2 per TC", True)):
    best = (np.inf, None)
    for tau in (0.3, 0.5, 1.0):
        v_ = run_glicko(tr, va, tau, per_tc).mean()
        if v_ < best[0]: best = (v_, tau)
    res[label] = run_glicko(tr, te, best[1], per_tc)
    print(f"  {label:16s} tau {best[1]:3.1f}  val {best[0]:.4f}  "
          f"TEST {res[label].mean():.4f}")
best = (np.inf, None)
for lam in (1.0, 3.0, 10.0):
    v_ = run_factor(tr, va, lam, scalar=True).mean()
    if v_ < best[0]: best = (v_, lam)
res["our scalar"] = run_factor(tr, te, best[1], scalar=True)
print(f"  {'our scalar':16s} ridge {best[1]:4.1f}  val {best[0]:.4f}  "
      f"TEST {res['our scalar'].mean():.4f}")
best = (np.inf, None)
for lam in (3.0, 10.0, 30.0):
    v_ = run_factor(tr, va, lam).mean()
    if v_ < best[0]: best = (v_, lam)
res["factor"] = run_factor(tr, te, best[1])
print(f"  {'factor':16s} ridge {best[1]:4.1f}  val {best[0]:.4f}  "
      f"TEST {res['factor'].mean():.4f}")

ids = np.random.default_rng(SEED+99).integers(0, len(te), (20000, len(te)))
def cmp(a, bname):
    d = (res[a][ids] - res[bname][ids]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a} vs {bname}: {res[a].mean()-res[bname].mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  P(better) {np.mean(d<0):.3f}")
print()
print("  --- the headline ---")
cmp("factor", "glicko2 per TC")
print("  --- decomposition: estimator vs structure ---")
cmp("our scalar", "glicko2 pooled")     # estimator effect
cmp("factor", "our scalar")             # structure effect
cmp("glicko2 pooled", "glicko2 per TC") # does stratification help THEM
pd.DataFrame(res).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"exp35_glicko2_seed{SEED}.csv"), index=False)
