"""exp31: the opposition-strength axis — giant-killers vs flat-track bullies.

Second candidate in the chess dimension search ("you give up too
easily"). The covariate is the signed opponent-strength gap, and it
passes the exogeneity requirement BY CONSTRUCTION: pairing hands every
player a spread of opponent strengths, so within-player variation is
guaranteed — the thing opening family lacked entirely.

The folk axis: some players over-perform against stronger opposition
(giant-killers) and others farm the weak (flat-track bullies), relative
to the rating model's own win curve.

DESIGN. Per-entrant covariate (each side of the same game sees the
mirrored gap): player i in a game vs opponent with site-rating gap
g = (elo_opp - elo_self)/400 carries x_i = [1, tanh(g)]. The FAIR
scalar baseline is not the bare scalar: it is scalar + a GLOBAL
curvature term (one shared tanh coefficient), so the factor arm's only
addition is PER-PLAYER curvature offsets, shrunk. Colour terms as in
exp30.

HONESTY NOTE, registered: site Elos encode history beyond the month
(information leak into both arms equally, via the shared global curve
and the offsets' covariate). The comparison factor-vs-baseline is
internally fair; absolute levels should not be compared with exp30's.

PRE-REGISTERED PREDICTIONS:
  P1. Identifiable: finite tuned offset ridge, factor beats
      scalar+globalcurve with CI excluding zero. Confidence LOWER than
      exp30 — the folk wisdom is noisier and Elo's curve already fits
      most of it.
  P2. The global curvature term itself is nonzero (the win curve
      deviates from the probit's implied slope in these pools).
  FALSIFICATION: offset ridge tunes to the scalar limit -> the
  giant-killer axis is not identifiable at one month of games, and the
  search moves to the next candidate (colour specialisation), NOT to a
  domain-level negative.

Run:  python experiments/exp31_chess_giant_killers.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map

CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
df = pd.read_parquet(CACHE)
df = df[df.result.isin(['1-0', '0-1'])]
df = df[pd.to_numeric(df.welo, errors='coerce').notna()
        & pd.to_numeric(df.belo, errors='coerce').notna()]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= 100].index)
df = df[df.white.isin(keep) & df.black.isin(keep)].reset_index(drop=True)
players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
Mp = len(players)
w = df.white.map(pidx).values; b = df.black.map(pidx).values
white_won = (df.result == '1-0').values
gap_w = np.tanh((pd.to_numeric(df.belo) - pd.to_numeric(df.welo)).values/400.0)
n = len(df)
print(f"{n} games, {Mp} players; |gap| deciles "
      + " ".join(f"{q:.2f}" for q in np.percentile(np.abs(gap_w), [10,50,90])))
rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
P = 2                       # [1, tanh(gap)] per entrant
NG = 2                      # white advantage + GLOBAL curvature

def build(profile, idx):
    width = (Mp*P if profile else Mp) + NG
    out = []
    for t in idx:
        Z = np.zeros((2, width))
        if profile:
            Z[0, w[t]*P] = 1; Z[0, w[t]*P+1] = gap_w[t]
            Z[1, b[t]*P] = 1; Z[1, b[t]*P+1] = -gap_w[t]
            g0 = Mp*P
        else:
            Z[0, w[t]] = 1; Z[1, b[t]] = 1
            g0 = Mp
        Z[0, g0] = 1.0                       # white advantage
        Z[0, g0+1] = gap_w[t]                # global curvature (white row)
        Z[1, g0+1] = -gap_w[t]
        out.append((Z, np.array([0, 1]) if white_won[t] else np.array([1, 0])))
    return out, width

def score(th, profile, idx):
    out = []
    for t in idx:
        if profile:
            mu_w = th[w[t]*P] + th[w[t]*P+1]*gap_w[t] + th[Mp*P] + th[Mp*P+1]*gap_w[t]
            mu_b = th[b[t]*P] - th[b[t]*P+1]*gap_w[t] - th[Mp*P+1]*gap_w[t]
        else:
            mu_w = th[w[t]] + th[Mp] + th[Mp+1]*gap_w[t]
            mu_b = th[b[t]] - th[Mp+1]*gap_w[t]
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

evs, ws_ = build(False, tr)
rv = np.ones(ws_); rv[Mp] = 0.1; rv[Mp+1] = 0.1
th_s = linear_ability_map(evs, ws_, ridge=rv)
base = score(th_s, False, te)
print(f"  scalar + global curve   TEST {base.mean():.4f}   "
      f"(global curvature coef {th_s[Mp+1]:+.3f})")

evf, wf_ = build(True, tr)
best = (np.inf, None, None)
for lam_off in (3.0, 10.0, 30.0, 100.0, 300.0):
    v = np.ones(wf_); v[np.arange(Mp)*P+1] = lam_off
    v[Mp*P] = 0.1; v[Mp*P+1] = 0.1
    th = linear_ability_map(evf, wf_, ridge=v)
    vv = score(th, True, va).mean()
    print(f"    offset ridge {lam_off:6.1f}  val {vv:.4f}")
    if vv < best[0]: best = (vv, lam_off, th)
factor = score(best[2], True, te)
edge = "  <-- SCALAR LIMIT: axis unidentifiable" if best[1] == 300.0 else ""
print(f"  factor (per-player)     TEST {factor.mean():.4f}  "
      f"(offset ridge {best[1]}){edge}")
ids = np.random.default_rng(1).integers(0, len(base), (20000, len(base)))
d = (factor[ids] - base[ids]).mean(1)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"  factor - baseline: {factor.mean()-base.mean():+.4f} "
      f"[{lo:+.4f},{hi:+.4f}]  P(better) {np.mean(d < 0):.3f}")
off = best[2][np.arange(Mp)*P+1]
print(f"  surviving curvature-offset sd: {off.std():.4f}")
pd.DataFrame({"base": base, "factor": factor}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp31_chess_gk.csv"), index=False)
