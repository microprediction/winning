"""exp38: does the Lichess result replicate on a modern month?

=====================================================================
VERDICT: NOT RUN. INFEASIBLE ON MODERN LICHESS, established by pre-run
diagnostics rather than by a null. The time-control covariate is
UNIDENTIFIABLE in 2024-01 at every threshold tried, for two
independent and compounding reasons.

(1) NO WITHIN-PLAYER CONTRAST. A factor rating over time control needs
players who play more than one control; otherwise the offsets are
confounded with the player's level and only the level is identified.

    threshold  players   games    bullet%   >=20 games in >=2 controls
        100      3,780    53,248    93.1%        11  ( 0.3%)
         50     19,665   289,714    80.7%        66  ( 0.3%)
         25     66,745   901,679    65.2%       347  ( 0.5%)
         10    194,843 2,072,055    51.7%       904  ( 0.5%)
    2013-01        632    59,399       --       327  (52.0%)

    Fifty-two percent in 2013 against half a percent now. Note also
    what the bullet column does: lowering the threshold makes the
    POPULATION more time-control diverse (93% -> 52% bullet) while
    individuals stay specialised. Between-player diversity without
    within-player contrast is exactly the signature that killed
    opening family in exp29 -- self-selection destroying the
    within-player covariate variation identification needs.

(2) MORE DATA WOULD MAKE IT WORSE, NOT BETTER. The obvious fix is a
bigger sample so players cross the >=20-in-2 bar. The data says the
opposite: contrast FALLS as activity rises. The >=100-game players are
the most active and also the most specialised -- 93.1% bullet and 0.3%
with contrast, the worst row in the table. On modern Lichess the more
someone plays, the more they specialise, because bullet is what lets
you play hundreds of games a month. Downloading the rest of the month
buys more of the least useful players.

The obvious escape -- restrict to players who DO mix controls -- fails
from the other side. Those players exist but barely play each other:

    904 multi-control players share only 2,839 games, obs/param 1.05
    3,860 (at >=10 in >=2)    11,007 games, obs/param 0.95
    2013 exp30: 632 players,  59,399 games, obs/param 30.9

    One observation per parameter. The time-control mix is finally
    healthy there (57% bullet / 42% blitz) and the design is still
    unfittable.

WHAT THIS IS AND IS NOT. This is NOT evidence against the 2013 result,
and the falsification clause below was written precisely to stop me
reading it that way. No arm was fitted, so nothing was measured about
the factor rating's accuracy. What was measured is that modern Lichess
cannot pose the question: the covariate has no within-player variation
to exploit. That is a covariate-level and data-level negative, in the
sense of exp29's opening-family failure, not a domain-level one.

The honest status of the headline is therefore unchanged and its
limitation is sharper than "one month of 2013": the result holds on a
population where players mixed time controls, and that population no
longer exists at scale on this site. Testing it on modern data needs
either a site whose players still mix controls, or a covariate that
modern players do not self-select into.

Everything below is the original registration, kept intact because the
predictions were made before the diagnostics were run.
=====================================================================

The largest hole in the chess result: everything so far is 2013-01,
639 players, an early-adopter population, and a rating system Lichess
has since changed. This replicates on 2024-01.

DATA. The month is ~32 GB compressed, so loader.py streams it with
early termination: the first 4,000,000 games, a contiguous opening
window of the month rather than a sample. 765,053 distinct players;
4,389 with >= 100 games, against 639 in 2013. The cache filename
records the cap so a truncated month cannot be mistaken for a whole
one.

    python research/chess/loader.py 2024-01 4000000

A DESIGN CHANGE FORCED BY THE DATA, not by preference. The time-control
mix has shifted drastically:

    2013-01   classical 34%   blitz 38%   bullet 27%
    2024-01   classical 0.6%  blitz 48%   bullet 37%

Classical has all but vanished (24,756 games of 4,000,000). Since the
2013 design used classical as the BASE category, keeping it would put
the model's reference level on a category with almost no data. The
base is therefore blitz -- the plurality -- with offsets for bullet
and classical. The classical offset is expected to be poorly
estimated; that is a fact about modern Lichess, and it is reported
rather than hidden by dropping the category.

    CORRECTION, found later and left here because the original claim
    was committed and is misleading. Those 2024 percentages are the
    RAW month. They are not the population this experiment analyses.
    After the >=100-game filter the ANALYSIS population is

        bullet 93.1%   blitz 6.9%   classical 0.01%

    because the threshold selects bullet specialists -- you can play
    hundreds of one-minute games in a month and not hundreds of
    classical ones. So "blitz is the plurality" was true of the month
    and false of the data, and the base category should have been
    bullet. This is moot given the VERDICT above, but the reasoning
    error is the point: I characterised the sample using a statistic
    computed before the sample was selected.

Note what this change does NOT do: it does not alter which games enter
the comparison, and every arm -- Glicko-2 pooled, Glicko-2 per time
control, our scalar, our factor -- sees exactly the same games and the
same split. Changing the reference level of a saturated set of
category offsets cannot favour an arm, because the fitted ability
differences are invariant to it.

THIS IS NOT A DENSITY-MATCHED REPLICATION, and pretending otherwise
would make the result uninterpretable. Measured before running:

                    total games   dense players   dense games   obs/param
    2013-01             121,332             639        59,399       30.92
    2024-01 (4M)      3,293,889           3,847        53,248        4.61

The dense-subgraph filter keeps **50.7%** of 2013's month and **1.6%**
of 2024's. 2013 Lichess was a small, tightly connected community where
the active players met each other repeatedly; by 2024 the site is so
large that even its most active players rarely meet twice. The
sparsity is in the PAIRING structure, not in activity, so no threshold
repairs it -- restricting to the top 639 players by volume makes it
worse, not better (obs/param 3.84), because the pool they are drawn
from is 6x larger.

Nor does more data repair it at any affordable scale. Across prefixes
of the cached month obs/param scales empirically as games^0.50, so the
FULL month (~95-100M games, about 30x this prefix) projects to roughly
25 -- still short of 30.92, and that is a 30x extrapolation from a fit
whose two smallest points are thin. Matching 30.92 outright projects
to 150-200M games, more than the month contains, but that figure
extrapolates ~45x beyond the measured range and should be read as an
order of magnitude rather than a number. The robust claim is the
weaker one: downloading the rest of the month would not close this
gap. The community grew far faster than any individual's game count,
and that is a permanent fact about modern Lichess, not a property of
this 4M prefix.

So this experiment tests the pooling claim in a regime ~7x sparser
than the one it was established in. That is a harder test, not a
lesser one, and it is the reason the predictions below are genuinely
uncertain rather than a formality.

PRE-REGISTERED PREDICTIONS, and two of my own results are in tension
here, which is why this is worth running:
  P1. The factor rating still beats glicko2-per-TC, CI excluding zero.
  P2. DIRECTION UNCERTAIN, stated as a fork rather than a guess. The
      pooling thesis (exp28e: advantage monotone in cell sparsity)
      predicts the margin WIDENS, because pooling pays most when data
      per cell is scarce, and this is the scarcest regime yet tested.
      But exp37 tested that prediction in chess and it FAILED -- the
      margin did not widen as players got sparser. I do not get to
      claim both. I predict the margin does NOT widen, siding with the
      measured chess evidence over the theory, and if it widens then
      exp37's null was the anomaly and the pooling story is stronger
      than the chess data has so far shown.
      The tension is real but NOT a straight contradiction, and the
      difference matters. exp37 lowered the threshold on one small
      month, which admits less active players while the community
      stays tightly connected; it moved obs/param only from 30.9 to
      15.3. This sits at 4.61 -- 3.3x beyond exp37's sparsest measured
      point, and sparse for a different reason (players rarely MEET
      rather than rarely play). exp37's null bounds the range where
      the margin is known flat; it does not extend to here.
      Working against both: classical has nearly vanished, so there is
      less time-control structure of any kind left to find.
  P3. The estimator/structure split still favours the estimator
      (2013: ~80% estimator, ~20% structure).

  FALSIFICATION, and the distinction that makes it meaningful: if the
  factor rating does not beat glicko2-per-TC, that is NOT automatically
  "the 2013 headline was an artifact." Sparsity and era are two
  different explanations and the decomposition separates them:

    - if `our scalar` still beats `glicko2 pooled` (the ESTIMATOR
      column survives) but `factor` no longer beats `our scalar` (the
      STRUCTURE column vanishes), the cause is sparsity plus the
      collapse of classical -- there is not enough per-player data to
      identify three parameters where one will do. The estimator claim
      stands; the factor claim is bounded to denser regimes.
    - if BOTH columns vanish, the 2013 result does not survive to the
      modern era and the headline must be restated as a 2013 finding.

  Reporting a sparsity failure as an era failure, or vice versa, is
  the specific error this paragraph exists to prevent.

Run:  python research/chess/exp38_modern_month.py
"""
from __future__ import annotations
import collections, importlib.util, os, sys, types
import numpy as np, pandas as pd
from scipy import sparse
from winning import race_probabilities
from winning.ratings.factor_ratings import fit_design_ratings as linear_ability_map


def _load_glicko():
    """Glicko-2 lives in src/winning/, a separate source tree from the
    installed package, so `import winning.glicko2` does not work. It is
    loaded by path. (A previous edit "simplified" this to a direct
    import; the shim is load-bearing.)"""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    roots = [os.path.join(repo, "src", "winning"),
             os.path.expanduser("~/github/winning/src/winning")]
    root = next((r for r in roots if os.path.isdir(r)), None)
    if root is None:
        raise ImportError(f"glicko2 source tree not found in {roots}")
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


Glicko2Rating = _load_glicko()
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser(
    "~/.cache/winning/lichess_2024_01_first4000000_headers.parquet")
TCS = ["blitz", "bullet", "classical"]      # blitz is the BASE now
MIN_GAMES = 100
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

if not os.path.exists(CACHE):
    raise SystemExit(
        f"missing {CACHE}\nrun: python research/chess/loader.py 2024-01 4000000")

df = pd.read_parquet(CACHE)
df = df[df.result.isin(["1-0", "0-1"])]
ev = df.event.str.lower()
tcn = np.where(ev.str.contains("bullet"), "bullet",
      np.where(ev.str.contains("blitz"), "blitz",
      np.where(ev.str.contains("classical"), "classical", "other")))
m = tcn != "other"
df = df[m].reset_index(drop=True); tcn = tcn[m]
cnt = pd.concat([df.white, df.black]).value_counts()
keep = set(cnt[cnt >= MIN_GAMES].index)
m2 = (df.white.isin(keep) & df.black.isin(keep)).values
df = df[m2].reset_index(drop=True); tcn = tcn[m2]
players = sorted(keep); pidx = {p: i for i, p in enumerate(players)}
Mp = len(players)
tci = pd.Series(tcn).map({t: i for i, t in enumerate(TCS)}).values
w = df.white.map(pidx).values; b = df.black.map(pidx).values
won = (df.result == "1-0").values
n = len(df)
P, NG = 3, 1 + len(TCS)          # defined BEFORE first use (was a bug)
print(f"2024-01 (first 4M games): {n} dense decisive games, {Mp} players")
print(f"  TC mix: {pd.Series(tcn).value_counts(normalize=True).round(4).to_dict()}")
_w = Mp * P + NG
print(f"  obs/param {n/_w:.2f} (2013 was 30.92) -- {30.92/(n/_w):.1f}x sparser;"
      f" this is NOT a density-matched replication")

# ---- the identifiability gate: run BEFORE fitting anything ----------
# A factor rating over time control needs WITHIN-player contrast. exp29
# established that a covariate players self-select into is
# unidentifiable however much data there is, so this is checked rather
# than assumed. exp30's 2013 population: 327 of 632 players (52%).
_bytc = collections.defaultdict(collections.Counter)
for _wp, _bp, _tt in zip(df.white.values, df.black.values, tcn):
    _bytc[_wp][_tt] += 1; _bytc[_bp][_tt] += 1
_contrast = [p for p, c in _bytc.items()
             if sum(1 for _k, v in c.items() if v >= 20) >= 2]
_frac = len(_contrast) / max(len(_bytc), 1)
print(f"  players with >=20 games in >=2 controls: {len(_contrast)}/{len(_bytc)}"
      f" ({_frac:.1%})   [2013: 327/632 = 52.0%]")
if _frac < 0.10:
    raise SystemExit(
        "\nSTOP: the time-control covariate is UNIDENTIFIABLE here.\n"
        f"Only {_frac:.1%} of players have contrast, against 52% in 2013.\n"
        "Fitting would produce a null that says nothing about the factor\n"
        "rating and everything about modern Lichess's self-selection --\n"
        "exp29's opening-family failure in a new costume. See the VERDICT\n"
        "block at the top of this file for the full diagnostics, including\n"
        "why more data makes this worse rather than better.\n"
        "NOT a refutation of the 2013 result; no arm was fitted.")
# ---------------------------------------------------------------------

rng = np.random.default_rng(SEED); perm = rng.permutation(n)
ntr, nva = int(n * .50), int(n * .25)
tr = np.sort(perm[:ntr])
va = np.sort(perm[ntr:ntr + nva])
te = np.sort(perm[ntr + nva:])
print(f"  split seed {SEED}: train {len(tr)} val {len(va)} TEST {len(te)}",
      flush=True)


def xvec(t):
    return np.array([1.0, float(tci[t] == 1), float(tci[t] == 2)])


def glicko(fit, ev_, tau, per_tc):
    S = ({c: Glicko2Rating(tau=tau) for c in range(len(TCS))} if per_tc
         else {0: Glicko2Rating(tau=tau)})
    for t in fit:
        S[tci[t] if per_tc else 0].observe(
            [players[w[t]], players[b[t]]], [1, 2] if won[t] else [2, 1], 1.0)
    out = []
    for t in ev_:
        p = np.asarray(S[tci[t] if per_tc else 0].win_probabilities(
            [players[w[t]], players[b[t]]]), dtype=float)
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if won[t] else p[1]))
    return np.array(out)


def ours(fit, ev_, lam, scalar=False):
    """Design rows are built SPARSE, and at this scale that is not an
    optimisation -- it is the difference between running and not.

    A dense (2, width) block per game costs
    len(fit) x 2 x width x 8 bytes. At 2013's 639 players that is
    width 1,921 and ~0.9 GB: survivable. Here width is 3 x 3,847 + 4 =
    11,545 and the training half alone is 4.92 GB, rebuilt for every
    point of the ridge grid, on a 17 GB machine.

    Each row has at most five non-zeros (three player columns, the
    intercept, one time-control offset), so the block is 0.035% dense
    and the same design costs 2.6 MB as CSR -- 1,924x smaller. Memory
    then scales with GAMES x NON-ZEROS rather than GAMES x PARAMETERS,
    which is the only form in which a rating over a large population
    is representable at all. fit_design_ratings accepts sparse Z and
    assembles one CSR internally, so nothing downstream changes.
    """
    pp = 1 if scalar else P
    width = Mp * pp + NG
    evs = []
    for t in fit:
        x = xvec(t)[:pp]
        rows = [0] * pp + [1] * pp
        cols = ([w[t] * pp + k for k in range(pp)]
                + [b[t] * pp + k for k in range(pp)])
        vals = list(x) + list(x)
        rows += [0, 0]
        cols += [Mp * pp, Mp * pp + 1 + tci[t]]
        vals += [1.0, 1.0]
        Z = sparse.csr_matrix((vals, (rows, cols)), shape=(2, width))
        evs.append((Z, np.array([0, 1]) if won[t] else np.array([1, 0])))
    v = np.full(width, 1.0)
    if not scalar:
        v[np.concatenate([np.arange(Mp) * P + 1, np.arange(Mp) * P + 2])] = lam
    v[Mp * pp] = 0.1; v[Mp * pp + 1:] = 1.0
    th = linear_ability_map(evs, width, ridge=v)
    out = []
    for t in ev_:
        x = xvec(t)[:pp]
        mu_w = th[w[t] * pp:w[t] * pp + pp] @ x + th[Mp * pp] + th[Mp * pp + 1 + tci[t]]
        mu_b = th[b[t] * pp:b[t] * pp + pp] @ x
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if won[t] else p[1]))
    return np.array(out)


res = {}
for lbl, per in (("glicko2 pooled", False), ("glicko2 per TC", True)):
    best = min(((glicko(tr, va, t, per).mean(), t) for t in (0.3, 0.5)))
    res[lbl] = glicko(tr, te, best[1], per)
    print(f"  {lbl:16s} TEST {res[lbl].mean():.4f}  (tau {best[1]})", flush=True)
# Grids are wider than 2013's. At obs/param 4.6 rather than 30.9 the
# optimal penalty must land much higher, and a grid that silently
# pins at its edge would understate every arm it constrains.
SCALAR_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
FACTOR_GRID = (3.0, 10.0, 30.0, 100.0, 300.0)
best = min(((ours(tr, va, l, True).mean(), l) for l in SCALAR_GRID))
res["our scalar"] = ours(tr, te, best[1], True)
edge = "  <-- GRID EDGE" if best[1] in (SCALAR_GRID[0], SCALAR_GRID[-1]) else ""
print(f"  {'our scalar':16s} TEST {res['our scalar'].mean():.4f}  "
      f"(ridge {best[1]}){edge}", flush=True)
best = min(((ours(tr, va, l).mean(), l) for l in FACTOR_GRID))
res["factor"] = ours(tr, te, best[1])
edge = "  <-- GRID EDGE" if best[1] in (FACTOR_GRID[0], FACTOR_GRID[-1]) else ""
print(f"  {'factor':16s} TEST {res['factor'].mean():.4f}  (ridge {best[1]})"
      f"{edge}", flush=True)

ids = np.random.default_rng(1).integers(0, len(te), (20000, len(te)))


def cmp(a, bb):
    d = (res[a][ids] - res[bb][ids]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a} vs {bb}: {res[a].mean()-res[bb].mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  P(better) {np.mean(d<0):.3f}")


print("\n  --- headline (P1): factor vs Lichess's deployed architecture ---")
cmp("factor", "glicko2 per TC")
print("  --- decomposition (P3) ---")
cmp("our scalar", "glicko2 pooled")      # estimator
cmp("factor", "our scalar")              # structure
cmp("glicko2 pooled", "glicko2 per TC")  # does stratification help Glicko-2?
d = res["factor"].mean() - res["glicko2 per TC"].mean()
print(f"\n  P2 wants |{d:+.4f}| < 0.0161 (2013's margin): "
      f"{'CONFIRMED' if abs(d) < 0.0161 else 'FAILED'}")
out = os.path.join(HERE, "results", f"exp38_modern_2024_seed{SEED}.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(res).to_csv(out, index=False)
print(f"  per-game losses -> {out}")
