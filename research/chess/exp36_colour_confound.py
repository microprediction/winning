"""exp36: is the 'estimator' advantage partly just modelling colour?

The first robustness check a referee would run. Our arms carry global
and per-time-control white-advantage terms; standard Glicko-2 models
colour NOT AT ALL. If a large slice of the -0.0206 estimator gap were
really "we know White scores ~54% and they do not", the README's
decomposition would be crediting structure-free bookkeeping to the
estimator.

RESULT: it is not the confound.

  our scalar + colour      0.6117
  our scalar, NO colour    0.6133
  glicko2 pooled           0.6323

  colour is worth to us            -0.0016
  estimator gap WITHOUT colour     -0.0190   <- like-for-like
  estimator gap WITH colour        -0.0206   (as published)

Colour accounts for 8% of the estimator advantage. Stripped entirely,
a colour-blind batch Thurstonian MAP fit still beats colour-blind
Glicko-2 by 0.019 on the same games. The published decomposition
stands; the honest footnote is that ~0.002 of the estimator column is
colour bookkeeping rather than estimation.

Run:  python research/chess/exp36_colour_confound.py
"""
import numpy as np, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(_HERE, 'exp35_vs_glicko2.py')).read()
head = src.split('res = {}')[0]
ns = {'__name__': 'prep',
      '__file__': os.path.join(_HERE, 'exp35_vs_glicko2.py')}
exec(compile(head, 'h', 'exec'), ns)
np, Mp, TCS = ns['np'], ns['Mp'], ns['TCS']
tci, w, b, white_won = ns['tci'], ns['w'], ns['b'], ns['white_won']
tr, va, te = ns['tr'], ns['va'], ns['te']
from winning.ratings.factor_ratings import fit_design_ratings
from winning import race_probabilities

def run(idx_fit, idx_eval, ridge, colour):
    """scalar ratings, with or without colour terms."""
    NG = (1 + len(TCS)) if colour else 0
    width = Mp + NG
    ev = []
    for t in idx_fit:
        Z = np.zeros((2, width))
        Z[0, w[t]] = 1; Z[1, b[t]] = 1
        if colour:
            Z[0, Mp] = 1.0; Z[0, Mp + 1 + tci[t]] = 1.0
        ev.append((Z, np.array([0,1]) if white_won[t] else np.array([1,0])))
    rv = np.full(width, ridge)
    if colour:
        rv[Mp] = 0.1; rv[Mp+1:] = 1.0
    th = fit_design_ratings(ev, width, ridge=rv)
    out = []
    for t in idx_eval:
        mu_w = th[w[t]] + ((th[Mp] + th[Mp+1+tci[t]]) if colour else 0.0)
        mu_b = th[b[t]]
        p = np.asarray(race_probabilities(-np.array([mu_w, mu_b]), D=np.ones(2)))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        out.append(-np.log(p[0] if white_won[t] else p[1]))
    return np.array(out)

res = {}
for label, colour in (("our scalar + colour", True),
                      ("our scalar, NO colour", False)):
    best = (np.inf, None)
    for r in (1.0, 3.0):
        v = run(tr, va, r, colour).mean()
        if v < best[0]: best = (v, r)
    res[label] = run(tr, te, best[1], colour)
    print(f"  {label:24s} TEST {res[label].mean():.4f}", flush=True)
print()
print(f"  glicko2 pooled (no colour, from exp35)   0.6323")
nc = res["our scalar, NO colour"].mean(); wc = res["our scalar + colour"].mean()
print(f"  colour is worth to us:        {wc - nc:+.4f}")
print(f"  estimator gap WITHOUT colour: {nc - 0.6323:+.4f}   "
      f"(like-for-like vs Glicko-2)")
print(f"  estimator gap WITH colour:    {wc - 0.6323:+.4f}   (as published)")
import pandas as pd
pd.DataFrame(res).to_csv(
    os.path.join(_HERE, 'results', 'exp36_colour.csv'), index=False)
