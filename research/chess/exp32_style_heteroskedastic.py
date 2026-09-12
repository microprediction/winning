"""exp32: project the factor rating into the engine's factor grammar.

Peter: "we could also project to factor models that the new winning
can handle." The fitted style vectors ARE loading rows. Performance
becomes x_i = d_i.x_c + v_i f_c + eps_i with v_i DERIVED from the
fitted time-control offsets (never free-fitted; exp02 showed free
loadings are hopeless), f_c a shared condition shock. That is exactly
race_probabilities(mu, V=...).

AT K=2 this makes a prediction the flat factor model cannot express:
a shared factor cancels in the margin unless loadings DIFFER, so the
margin noise is 1 + s^2 (v_w - v_b)^2. Style-opposite pairings should
be MORE RANDOM than the rating gap predicts. Never tested, testable
today.

ARMS (same fitted mu from exp30's tuned factor model throughout; only
the PRICING changes):
  flat        D = 1 for both entrants (exp30's pricing)
  hetero(s)   engine-priced with per-entrant loading v_i = s * (that
              player's offset for the game's time control, centred),
              s swept on validation; s=0 recovers flat exactly.

PRE-REGISTERED PREDICTIONS:
  P1. The tuned s is nonzero and the hetero arm improves on flat with
      CI excluding zero -- CONFIDENCE LOW; the sign could genuinely go
      the other way (style clashes might be MORE decisive, not less).
  P2. If P1 holds, the improvement concentrates in high style-gap
      games (mechanical, but confirms the pathway).
  FALSIFICATION: tuned s = 0. Then the projection adds nothing at
  K=2 and its value waits on K-way fields, stated as such.

Run:  python experiments/exp32_style_heteroskedastic.py
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
rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
NG = 1 + len(TCS); P = 3

def xvec(t):
    return np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])

# refit exp30's tuned factor model (offset ridge 10.0) -- deterministic
evf = []
width = Mp*P + NG
for t in tr:
    x = xvec(t)
    Z = np.zeros((2, width))
    Z[0, w[t]*P:w[t]*P+P] = x; Z[1, b[t]*P:b[t]*P+P] = x
    Z[0, Mp*P] = 1.0; Z[0, Mp*P+1+tci[t]] = 1.0
    evf.append((Z, np.array([0, 1]) if white_won[t] else np.array([1, 0])))
v = np.full(width, 1.0)
off_cols = np.concatenate([np.arange(Mp)*P+1, np.arange(Mp)*P+2])
v[off_cols] = 10.0
v[Mp*P] = 0.1; v[Mp*P+1:] = 1.0
th = linear_ability_map(evf, width, ridge=v)
print(f"refit done: {n} games, {Mp} players")

def game_style(t, i):
    """the offset THIS game's time control activates for player i,
    centred so classical-base players are not mechanically style-0"""
    if tci[t] == 2: return th[i*P+1]
    if tci[t] == 1: return th[i*P+2]
    return 0.0

def loss(idx, s):
    out, gaps = [], []
    for t in idx:
        x = xvec(t)
        mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+tci[t]]
        mu_b = th[b[t]*P:b[t]*P+P] @ x
        vw = s * game_style(t, w[t]); vb = s * game_style(t, b[t])
        if s == 0.0:
            p = np.asarray(race_probabilities(
                -np.array([mu_w, mu_b]), D=np.ones(2)))
        else:
            V = np.array([[vw], [vb]])
            p = np.asarray(race_probabilities(
                -np.array([mu_w, mu_b]), V=V, D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
        gaps.append(abs(vw - vb))
    return np.array(out), np.array(gaps)

flat_va, _ = loss(va, 0.0)
print(f"  flat (s=0)      val {flat_va.mean():.4f}")
best = (flat_va.mean(), 0.0)
for s in (0.5, 1.0, 2.0, 4.0):
    lv, _ = loss(va, s)
    print(f"  hetero s={s:3.1f}    val {lv.mean():.4f}")
    if lv.mean() < best[0]: best = (lv.mean(), s)
print(f"\ntuned s = {best[1]}")
flat_te, gap0 = loss(te, 0.0)
het_te, gap = loss(te, best[1] if best[1] > 0 else 1.0)
if best[1] == 0.0:
    print("FALSIFIED at K=2: tuned s = 0; the projection's value waits "
          "on K-way fields.")
    print(f"  (forced s=1 test diff, for the record: "
          f"{het_te.mean()-flat_te.mean():+.4f})")
else:
    d = het_te - flat_te
    ids = np.random.default_rng(1).integers(0, len(d), (20000, len(d)))
    bs = d[ids].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"hetero - flat on TEST: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"P(better) {np.mean(bs < 0):.3f}")
    med = np.median(gap[gap > 0]) if (gap > 0).any() else 0
    for lab, msk in (("style gap > median", gap > med),
                     ("style gap <= median", gap <= med)):
        if msk.sum() < 100: continue
        dd = d[msk]
        bs2 = dd[np.random.default_rng(2).integers(0, len(dd),
                                                   (20000, len(dd)))].mean(1)
        print(f"  {lab:22s} n={int(msk.sum()):6d}  {dd.mean():+.4f} "
              f"[{np.percentile(bs2,2.5):+.4f},{np.percentile(bs2,97.5):+.4f}]")
pd.DataFrame({"flat": flat_te, "hetero": het_te, "style_gap": gap}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp32_hetero.csv"),
    index=False)
print("per-game losses written to results/exp32_hetero.csv")
