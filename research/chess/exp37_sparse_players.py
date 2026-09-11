"""exp37: does pooling help sparse players most? The mechanism's own prediction.

Every chess result so far keeps only players with >= 100 games in the
month -- the dense subpopulation, where rating is easiest. Lichess has
to rate everyone, including the player with nine games. So the
threshold is a robustness question, but it is more than that: it is
the one place the pooling MECHANISM makes a falsifiable prediction
rather than a hedge.

If the factor rating wins because it POOLS -- level from all of a
player's games, only the offsets specialising -- then its advantage
should be LARGEST where data per player is scarcest, because that is
where stratification and per-player estimation starve and shrinkage
pays. If instead the margin narrows as the threshold drops, the win is
coming from somewhere else and the pooling story is wrong.

DESIGN. Rerun the four-arm comparison at thresholds 100, 50, 25, 10,
reporting the margin over Lichess's architecture and, separately,
within the SPARSE players at each threshold (those with fewer than 100
games, i.e. the players the earlier runs excluded). Same protocol
throughout; Glicko-2's tau and our ridge re-tuned at each threshold,
since the right amount of shrinkage should itself change with density.

PRE-REGISTERED PREDICTIONS:
  P1. The factor margin over glicko2-per-TC WIDENS monotonically as
      the threshold drops.
  P2. Restricted to sparse players, the margin is larger still than
      the all-player margin at the same threshold.
  P3. The tuned offset ridge RISES as the threshold drops: less data
      per player means more shrinkage is correct.
  FALSIFICATION: if the margin narrows as the threshold drops, the
  pooling explanation is wrong and the README's mechanism section
  needs rewriting, not just qualifying.

Run:  python research/chess/exp37_sparse_players.py
"""
from __future__ import annotations
import importlib.util, os, sys, types
import numpy as np, pandas as pd
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/winning/lichess_2013_01_headers.parquet")
TCS = ['classical', 'blitz', 'bullet']
THRESHOLDS = (100, 50, 25, 10)

def _glicko():
    repo = os.path.dirname(os.path.dirname(_HERE))
    root = next((r for r in (os.path.join(repo, "src", "winning"),
                             os.path.expanduser("~/github/winning/src/winning"))
                 if os.path.isdir(r)), None)
    pkg = types.ModuleType("wsrc"); pkg.__path__ = [root]
    sys.modules["wsrc"] = pkg
    for n in ("ratingsystem", "glicko2"):
        spec = importlib.util.spec_from_file_location(
            f"wsrc.{n}", os.path.join(root, n + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"wsrc.{n}"] = m; setattr(pkg, n, m)
        spec.loader.exec_module(m)
    from wsrc.glicko2 import Glicko2Rating
    return Glicko2Rating

Glicko2Rating = _glicko()
raw = pd.read_parquet(CACHE)
raw = raw[raw.result.isin(['1-0', '0-1'])]
_ev = raw.event.str.lower()
_tc = np.where(_ev.str.contains('bullet'), 'bullet',
      np.where(_ev.str.contains('blitz'), 'blitz',
      np.where(_ev.str.contains('classical'), 'classical', 'other')))
raw = raw[_tc != 'other'].reset_index(drop=True); _tc = _tc[_tc != 'other']

rows = []
for TH in THRESHOLDS:
    cnt = pd.concat([raw.white, raw.black]).value_counts()
    keep = set(cnt[cnt >= TH].index)
    mask = (raw.white.isin(keep) & raw.black.isin(keep)).values
    df = raw[mask].reset_index(drop=True); tcn = _tc[mask]
    players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
    Mp = len(players)
    tci = pd.Series(tcn).map({t: i for i, t in enumerate(TCS)}).values
    w = df.white.map(pidx).values; b = df.black.map(pidx).values
    won = (df.result == '1-0').values
    ngames = np.array([cnt[p] for p in players])
    n = len(df)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr, nva = int(n*.50), int(n*.25)
    tr, va, te = np.sort(perm[:ntr]), np.sort(perm[ntr:ntr+nva]), np.sort(perm[ntr+nva:])
    print(f"\n=== threshold {TH}: {n} games, {Mp} players ===", flush=True)

    def glicko(fit, ev_, tau):
        S = {c: Glicko2Rating(tau=tau) for c in range(len(TCS))}
        for t in fit:
            S[tci[t]].observe([players[w[t]], players[b[t]]],
                              [1, 2] if won[t] else [2, 1], 1.0)
        out = []
        for t in ev_:
            p = np.asarray(S[tci[t]].win_probabilities(
                [players[w[t]], players[b[t]]]), dtype=float)
            p = np.clip(p, 1e-12, None); p /= p.sum()
            out.append(-np.log(p[0] if won[t] else p[1]))
        return np.array(out)

    def factor(fit, ev_, lam):
        P, NG = 3, 1 + len(TCS); width = Mp*P + NG
        evs = []
        for t in fit:
            x = np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])
            Z = np.zeros((2, width))
            Z[0, w[t]*P:w[t]*P+P] = x; Z[1, b[t]*P:b[t]*P+P] = x
            Z[0, Mp*P] = 1.0; Z[0, Mp*P+1+tci[t]] = 1.0
            evs.append((Z, np.array([0,1]) if won[t] else np.array([1,0])))
        v = np.full(width, 1.0)
        v[np.concatenate([np.arange(Mp)*P+1, np.arange(Mp)*P+2])] = lam
        v[Mp*P] = 0.1; v[Mp*P+1:] = 1.0
        th = fit_design_ratings(evs, width, ridge=v)
        out = []
        for t in ev_:
            x = np.array([1.0, float(tci[t] == 2), float(tci[t] == 1)])
            mu_w = th[w[t]*P:w[t]*P+P] @ x + th[Mp*P] + th[Mp*P+1+tci[t]]
            mu_b = th[b[t]*P:b[t]*P+P] @ x
            p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]),
                                              D=np.ones(2)))
            p = np.clip(p, 1e-12, None); p /= p.sum()
            out.append(-np.log(p[0] if won[t] else p[1]))
        return np.array(out)

    bg = min(((glicko(tr, va, t).mean(), t) for t in (0.3, 0.5)))
    g = glicko(tr, te, bg[1])
    bf = min(((factor(tr, va, l).mean(), l) for l in (3.0, 10.0, 30.0)))
    f = factor(tr, te, bf[1])
    d = f - g
    ids = np.random.default_rng(5).integers(0, len(d), (20000, len(d)))
    bs = d[ids].mean(1)
    sparse = np.array([(ngames[w[t]] < 100) or (ngames[b[t]] < 100)
                       for t in te])
    print(f"  glicko2 per TC {g.mean():.4f}   factor {f.mean():.4f}  "
          f"(ridge {bf[1]})")
    print(f"  ALL      margin {d.mean():+.4f} "
          f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
    line = {"threshold": TH, "players": Mp, "games": n,
            "margin_all": d.mean(), "ridge": bf[1]}
    if sparse.sum() > 200:
        ds = d[sparse]
        bs2 = ds[np.random.default_rng(6).integers(0, len(ds),
                                                   (20000, len(ds)))].mean(1)
        print(f"  SPARSE   margin {ds.mean():+.4f} "
              f"[{np.percentile(bs2,2.5):+.4f},{np.percentile(bs2,97.5):+.4f}]"
              f"   n={int(sparse.sum())}")
        line["margin_sparse"] = ds.mean()
    rows.append(line)

out = pd.DataFrame(rows)
print("\n=== P1: does the margin widen as the threshold drops? ===")
print(out.to_string(index=False))
out.to_csv(os.path.join(_HERE, "results", "exp37_sparse.csv"), index=False)
