"""exp29: the positional/tactical axis — factor ratings in chess.

Peter's conjecture: the factor-rating result transfers to chess with
opening type as the observed covariate, and the axis players actually
differ on is POSITIONAL vs TACTICAL. His registered caution: openings
are not exchangeable — they differ in colour advantage and character —
so we must "condition on equal games of different types", which here
becomes SHARED opening-family terms that all players carry, so the
per-player offsets can only pick up differential skill, never the
opening's intrinsic character.

DATA. Lichess open database, 2013-01 (headers only): 121,332 rated
games; 59,403 decisive games where both players have >=100 games in the
month; 639 players. Draw rate 3.3%. ECO codes on every game.

COVARIATE. tactical = ECO family in {B, C} (1.e4 worlds: open and
semi-open games) vs {A, D, E} (flank, closed, Indian: positional).
Crude, stated as crude, refinable. Both players face the SAME opening,
exactly as both models face the same prompt in Arena.

DESIGN. mu difference in a game = (d_w - d_b)·x + colour terms, where
x = [1, tactical]; colour terms are a global white-advantage column
plus a per-ECO-family white-advantage deviation (SHARED by all players)
— the operational form of "equal after 4 moves". Arms:

  scalar+colour   x = [1]        + shared colour terms
  factor+colour   x = [1, tac]   + shared colour terms

Identical estimator, ridge tuned on validation, scored window untouched.

PRE-REGISTERED PREDICTIONS:
  P1. The axis is identifiable: factor beats scalar+colour with a CI
      excluding zero. The channel is Arena-sized and the folk wisdom
      (e4 players vs d4 players) is a century old.
  P2. The gain is LARGER than Arena's -0.0016: 639 club players vary
      more in style than 53 frontier models curated for all-roundedness.
  P3. The gain concentrates on games between players whose fitted
      tactical offsets DIFFER most (post-hoc diagnostic).
  FALSIFICATION: if factor does not beat scalar+colour, the transfer
  fails and the write-up says so.

Run:  python experiments/exp29_chess_openings.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map

CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
MIN_GAMES = 100
FAMS = ['A', 'B', 'C', 'D', 'E']

df = pd.read_parquet(CACHE)
df = df[df.result.isin(['1-0', '0-1'])]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= MIN_GAMES].index)
df = df[df.white.isin(keep) & df.black.isin(keep)].reset_index(drop=True)
players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
Mp = len(players)
fam = df.eco.fillna('?').str[0].where(df.eco.fillna('?').str[0].isin(FAMS), 'A')
fidx = fam.map({f: i for i, f in enumerate(FAMS)}).values
tac = fam.isin(['B', 'C']).values.astype(float)
w = df.white.map(pidx).values; b = df.black.map(pidx).values
white_won = (df.result == '1-0').values
n = len(df)
print(f"{n} decisive dense games, {Mp} players; tactical share {tac.mean():.3f}")

rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr, nva = int(n*.50), int(n*.25)
tr, va, te = perm[:ntr], perm[ntr:ntr+nva], perm[ntr+nva:]
NG = 1 + len(FAMS)          # global white-adv + per-family white-adv deviation

def build(profile, idx):
    P = 2 if profile else 1
    width = Mp*P + NG
    ev = []
    for t in idx:
        x = np.array([1.0, tac[t]])[:P]
        Z = np.zeros((2, width))
        Z[0, w[t]*P:w[t]*P+P] = x
        Z[1, b[t]*P:b[t]*P+P] = x
        Z[0, Mp*P] = 1.0                    # white advantage (white row only)
        Z[0, Mp*P + 1 + fidx[t]] = 1.0      # family colour deviation
        ev.append((Z, np.array([0, 1]) if white_won[t] else np.array([1, 0])))
    return ev, width, P

def ridge_vec(width, P, lam):
    v = np.full(width, lam)
    v[Mp*P] = 0.1                            # colour terms lightly penalised
    v[Mp*P+1:] = 1.0
    return v

def score(th, profile, idx):
    P = 2 if profile else 1
    out = []
    for t in idx:
        x = np.array([1.0, tac[t]])[:P]
        mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+fidx[t]]
        mu_b = th[b[t]*P:b[t]*P+P] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

results = {}
for label, profile in (("scalar+colour", False), ("factor+colour", True)):
    ev, width, P = build(profile, tr)
    best = (np.inf, None, None)
    for lam in (1.0, 3.0, 10.0):
        th = linear_ability_map(ev, width, ridge=ridge_vec(width, P, lam))
        v = score(th, profile, va).mean()
        if v < best[0]:
            best = (v, lam, th)
    lt = score(best[2], profile, te)
    results[label] = (lt, best[2])
    print(f"  {label:14s} ridge {best[1]:4.1f}  val {best[0]:.4f}  "
          f"TEST {lt.mean():.4f}")

base, factor = results["scalar+colour"][0], results["factor+colour"][0]
ids = np.random.default_rng(1).integers(0, len(base), (20000, len(base)))
d = (factor[ids] - base[ids]).mean(1)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nfactor - scalar: {factor.mean()-base.mean():+.4f} "
      f"[{lo:+.4f},{hi:+.4f}]  P(better) {np.mean(d < 0):.3f}")

# P3 diagnostic: split scored games by fitted tactical-offset gap
th_f = results["factor+colour"][1]
tacoff = th_f[np.arange(Mp)*2 + 1]
gap = np.abs(tacoff[w[te]] - tacoff[b[te]])
med = np.median(gap)
for lab, msk in (("style gap > median", gap > med),
                 ("style gap < median", gap <= med)):
    dd = factor[msk] - base[msk]
    bs = dd[np.random.default_rng(2).integers(0, len(dd), (20000, len(dd)))].mean(1)
    print(f"  {lab:22s} n={int(msk.sum()):6d}  {dd.mean():+.4f} "
          f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
print(f"\ntactical-offset spread across players: sd {tacoff.std():.3f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp29_chess.csv")
pd.DataFrame({"scalar": base, "factor": factor,
              "style_gap": gap}).to_csv(out, index=False)
print(f"per-game losses written to {out}")
