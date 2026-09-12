"""exp29b: separate the level ridge from the offset ridge.

exp29's factor arm lost (+0.0037) with damage concentrated where the
fitted style offsets were largest -- the signature of noise offsets.
But the harness had a defect: ONE shared ridge covered both player
levels and tactical offsets, so the tuner could not express "levels
free, offsets shrunk hard". Since the factor arm equals the scalar in
the offset-ridge -> infinity limit, a fair sweep must include that
limit. If the tuned offset ridge is effectively infinite, the
positional/tactical axis is genuinely unidentifiable in this data; if
a finite value wins with a gain, exp29's loss was the harness.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    'e29', os.path.join(os.path.dirname(__file__), 'exp29_chess_openings.py'))
# import module namespace WITHOUT running its main sweep: reuse by copy
CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map
FAMS = ['A','B','C','D','E']
df = pd.read_parquet(CACHE)
df = df[df.result.isin(['1-0','0-1'])]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= 100].index)
df = df[df.white.isin(keep) & df.black.isin(keep)].reset_index(drop=True)
players = sorted(keep); pidx = {p:i for i,p in enumerate(players)}
Mp = len(players)
fam = df.eco.fillna('?').str[0].where(df.eco.fillna('?').str[0].isin(FAMS),'A')
fidx = fam.map({f:i for i,f in enumerate(FAMS)}).values
tac = fam.isin(['B','C']).values.astype(float)
w = df.white.map(pidx).values; b = df.black.map(pidx).values
white_won = (df.result=='1-0').values
n = len(df)
rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
NG = 1 + len(FAMS)
P = 2
width = Mp*P + NG

ev = []
for t in tr:
    x = np.array([1.0, tac[t]])
    Z = np.zeros((2, width))
    Z[0, w[t]*P:w[t]*P+P] = x; Z[1, b[t]*P:b[t]*P+P] = x
    Z[0, Mp*P] = 1.0; Z[0, Mp*P+1+fidx[t]] = 1.0
    ev.append((Z, np.array([0,1]) if white_won[t] else np.array([1,0])))

def score(th, idx):
    out = []
    for t in idx:
        x = np.array([1.0, tac[t]])
        mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+fidx[t]]
        mu_b = th[b[t]*P:b[t]*P+P] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

def rvec(lam_level, lam_off):
    v = np.full(width, lam_level)
    v[np.arange(Mp)*P + 1] = lam_off        # tactical offsets only
    v[Mp*P] = 0.1; v[Mp*P+1:] = 1.0
    return v

print("level ridge 1.0 (exp29's tuned value); offset ridge swept to the "
      "scalar limit:")
best = (np.inf, None, None)
for lam_off in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
    th = linear_ability_map(ev, width, ridge=rvec(1.0, lam_off))
    v = score(th, va).mean()
    tag = " <- val best so far" if v < best[0] else ""
    print(f"  offset ridge {lam_off:7.1f}  val {v:.4f}{tag}")
    if v < best[0]: best = (v, lam_off, th)
lt = score(best[2], te)
print(f"\ntuned offset ridge {best[1]}  TEST {lt.mean():.4f}")
base = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp29_chess.csv")).scalar.values
d = lt - base
ids = np.random.default_rng(1).integers(0, len(d), (20000, len(d)))
bs = d[ids].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"factor(split ridge) - scalar: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
      f"P(better) {np.mean(bs<0):.3f}")
tacoff = best[2][np.arange(Mp)*P + 1]
print(f"surviving tactical-offset spread: sd {tacoff.std():.4f}")
pd.DataFrame({"factor_split": lt}).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp29b_chess.csv"), index=False)
