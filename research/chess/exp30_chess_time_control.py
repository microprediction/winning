"""exp30: Peter's dimension — time control. And the industry contest.

exp29 concluded "chess fails" from ONE covariate (opening family) that
the exogeneity diagnostic showed was ~deterministically self-selected.
Peter's correction: "chess hasn't failed... you just didn't pick the
dimensions. try time control." Time control is a stronger candidate on
every axis: the folk wisdom is stronger (bullet specialists are a
recognised type), and Lichess itself maintains SEPARATE ratings per
time control — i.e. the industry's live practice is STRATIFICATION,
which makes this exp28e transplanted to chess: does partial pooling
beat what Lichess actually does?

DIAGNOSTIC FIRST (the exp29 lesson): 327 of 632 dense players have
>=20 games in >=2 time controls; max-TC share median 0.81 against the
opening analogue's 0.98. Half the population carries identification;
shrinkage covers the rest.

ARMS (identical estimator; colour terms shared: global white advantage
plus per-TC white-advantage deviation):
  scalar+colour      one number per player
  factor+colour      x = [1, bullet, blitz], classical the base;
                     level ridge fixed at exp29's tuned 1.0, OFFSET
                     ridge swept to the scalar limit (exp29b's lesson
                     baked in, not retrofitted)
  stratified-by-TC   a separate scalar rating per time control —
                     Lichess's actual practice

PRE-REGISTERED PREDICTIONS:
  P1. Identifiable: factor beats scalar, CI excluding zero, tuned
      offset ridge finite.
  P2. Factor beats STRATIFIED — the pooling story — with the advantage
      concentrated in player-TC cells with few games.
  P3. The bullet offset spread exceeds the blitz offset spread: bullet
      is the most distinct skill.
  FALSIFICATION: offset ridge tuning to the top of its grid (the
  scalar limit) means time control fails like openings did, and the
  chess-transfer negative stands over Peter's objection.

Run:  python experiments/exp30_chess_time_control.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map

CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
TCS = ['classical', 'blitz', 'bullet']

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
print(f"{n} dense decisive games, {Mp} players; TC mix "
      f"{pd.Series(tcname).value_counts(normalize=True).round(3).to_dict()}")

rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
NG = 1 + len(TCS)                 # white adv + per-TC white-adv deviation
P = 3                             # [1, bullet, blitz]

def xvec(t):
    return np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])

def build_factor(idx):
    width = Mp*P + NG
    out = []
    for t in idx:
        x = xvec(t)
        Z = np.zeros((2, width))
        Z[0, w[t]*P:w[t]*P+P] = x; Z[1, b[t]*P:b[t]*P+P] = x
        Z[0, Mp*P] = 1.0; Z[0, Mp*P+1+tci[t]] = 1.0
        out.append((Z, np.array([0, 1]) if white_won[t] else np.array([1, 0])))
    return out, width

def build_scalar(idx):
    width = Mp + NG
    out = []
    for t in idx:
        Z = np.zeros((2, width))
        Z[0, w[t]] = 1; Z[1, b[t]] = 1
        Z[0, Mp] = 1.0; Z[0, Mp+1+tci[t]] = 1.0
        out.append((Z, np.array([0, 1]) if white_won[t] else np.array([1, 0])))
    return out, width

def score_factor(th, idx):
    out = []
    for t in idx:
        x = xvec(t)
        mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+tci[t]]
        mu_b = th[b[t]*P:b[t]*P+P] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

def score_scalar(th, idx):
    out = []
    for t in idx:
        mu_w = th[w[t]] + th[Mp] + th[Mp+1+tci[t]]
        mu_b = th[b[t]]
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

# ---- scalar+colour ----
evs, ws_ = build_scalar(tr)
rv = np.full(ws_, 1.0); rv[Mp] = 0.1; rv[Mp+1:] = 1.0
th_s = linear_ability_map(evs, ws_, ridge=rv)
base = score_scalar(th_s, te)
print(f"  scalar+colour      TEST {base.mean():.4f}")

# ---- factor+colour, offset ridge swept to the scalar limit ----
evf, wf_ = build_factor(tr)
best = (np.inf, None, None)
for lam_off in (3.0, 10.0, 30.0, 100.0, 300.0):
    v = np.full(wf_, 1.0)
    off_cols = np.concatenate([np.arange(Mp)*P+1, np.arange(Mp)*P+2])
    v[off_cols] = lam_off
    v[Mp*P] = 0.1; v[Mp*P+1:] = 1.0
    th = linear_ability_map(evf, wf_, ridge=v)
    vv = score_factor(th, va).mean()
    print(f"    offset ridge {lam_off:6.1f}  val {vv:.4f}")
    if vv < best[0]:
        best = (vv, lam_off, th)
factor = score_factor(best[2], te)
edge = "  <-- SCALAR LIMIT: axis unidentifiable" if best[1] == 300.0 else ""
print(f"  factor+colour      TEST {factor.mean():.4f}  (offset ridge {best[1]}){edge}")

# ---- stratified per TC: Lichess practice ----
strat = np.zeros(len(te))
te_pos = {t: i for i, t in enumerate(te)}
for c in range(len(TCS)):
    tr_c = tr[tci[tr] == c]; te_c = te[tci[te] == c]
    evc, wc_ = build_scalar(tr_c)
    rvc = np.full(wc_, 1.0); rvc[Mp] = 0.1; rvc[Mp+1:] = 1.0
    th_c = linear_ability_map(evc, wc_, ridge=rvc)
    sc = score_scalar(th_c, te_c)
    for t, v_ in zip(te_c, sc):
        strat[te_pos[t]] = v_
print(f"  stratified-by-TC   TEST {strat.mean():.4f}   (Lichess practice)")

ids = np.random.default_rng(1).integers(0, len(base), (20000, len(base)))
for name, arm in (("factor vs scalar", factor), ("stratified vs scalar", strat)):
    d = (arm[ids] - base[ids]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {name:22s} {arm.mean()-base.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"P(better) {np.mean(d < 0):.3f}")
d = (factor[ids] - strat[ids]).mean(1)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"  factor vs STRATIFIED   {factor.mean()-strat.mean():+.4f} "
      f"[{lo:+.4f},{hi:+.4f}]  P(better) {np.mean(d < 0):.3f}")

th_f = best[2]
bul = th_f[np.arange(Mp)*P+1]; bli = th_f[np.arange(Mp)*P+2]
print(f"\n  offset spreads: bullet sd {bul.std():.4f}   blitz sd {bli.std():.4f}"
      f"   (P3: bullet > blitz)")
pd.DataFrame({"scalar": base, "factor": factor, "strat": strat,
              "tc": [TCS[c] for c in tci[te]]}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp30_chess_tc.csv"),
    index=False)
print("per-game losses written to results/exp30_chess_tc.csv")
