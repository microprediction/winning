"""Fishing (mlogit's vignette dataset, J=4, T=1182): fit the python
reference MNProbit and emit the arrays + fitted theta + referee
likelihood for the cross-language table. The dataset is fetched from
R's mlogit package at run time (not vendored):

  Rscript -e 'data(Fishing, package="mlogit"); write.csv(Fishing, "fishing.csv", row.names=FALSE)'
  python julia/MultinomialProbit/benchmark/fishing_python.py fishing.csv out.json
"""
import json
import sys
import time

import numpy as np

from winning.mnprobit import MNProbit


def load(path):
    rows = [ln.strip().split(",") for ln in open(path).read().splitlines()]
    hdr = [h.strip('"') for h in rows[0]]
    alts = ["beach", "boat", "charter", "pier"]
    pi = {a: hdr.index(f"price.{a}") for a in alts}
    ci = {a: hdr.index(f"catch.{a}") for a in alts}
    mi = hdr.index("mode")
    T = len(rows) - 1
    X = np.zeros((T, 4, 2))
    choice = np.zeros(T, dtype=int)
    for t, r in enumerate(rows[1:]):
        for j, a in enumerate(alts):
            X[t, j, 0] = float(r[pi[a]])
            X[t, j, 1] = float(r[ci[a]])
        choice[t] = alts.index(r[mi].strip('"'))
    return X, choice


def main(csv_path, out_path):
    X, choice = load(csv_path)
    t0 = time.perf_counter()
    m = MNProbit(X, choice, intercepts=True, r=2).fit()
    dt = time.perf_counter() - t0
    out = {
        "X": X.tolist(), "choice": choice.tolist(),
        "python": {"theta": m.theta_.tolist(),
                   "loglik_referee": m.loglik_,
                   "loglik_se": m.loglik_se_,
                   "boundary": m.boundary_, "seconds": dt},
    }
    json.dump(out, open(out_path, "w"))
    print(f"python exact: referee logLik {m.loglik_:.2f} "
          f"+- {m.loglik_se_:.2f} in {dt:.1f}s "
          f"(boundary={m.boundary_})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
