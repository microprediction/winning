"""Are the committed vectors still a description of the python reference?

The ports are compared against parity/vectors.json, so a python change that
moves a scenario without regenerating it leaves them compared against
yesterday's answers -- eleven scenarios were stale that way when this was
written, and both port checkers passed throughout.

Byte-equality is the wrong test: regenerating on another platform moves the
iterative scenarios (inversions, polish, loc-scale solvers) in the last few
digits, because BLAS reduction order differs. So each scenario is compared
against the committed value at the SAME tolerance the ports must meet. If
python has drifted further than the ports are allowed to, the file is stale
in the only sense that matters.

    python parity/check_vectors.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_vectors import build, make_inputs           # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "vectors.json")) as fh:
        committed = json.load(fh)
    fresh = build(make_inputs())

    missing = sorted(set(committed["scenarios"]) - set(fresh))
    added = sorted(set(fresh) - set(committed["scenarios"]))
    fails = []
    for name, got in fresh.items():
        if name not in committed["scenarios"]:
            continue
        want = committed["scenarios"][name]
        a = np.asarray(want["value"], dtype=float)
        b = np.asarray(got["value"], dtype=float)
        if a.shape != b.shape:
            fails.append((name, "shape", a.shape, b.shape))
            continue
        d = float(np.abs(a - b).max()) if a.size else 0.0
        tol = float(want["tol"])
        print(f"{'ok  ' if d <= tol else 'STALE':5s} {name:28s} "
              f"max|committed - fresh| {d:.3e}  (tol {tol:g})")
        if d > tol:
            fails.append((name, d, tol, None))

    if added:
        print(f"\nscenarios generated but not committed: {added}")
    if missing:
        print(f"\nscenarios committed but no longer generated: {missing}")
    if fails or added or missing:
        print("\nparity/vectors.json is stale. Regenerate it with\n"
              "    WINNING_PURE=1 python parity/gen_vectors.py\n"
              "and check both ports before committing the result.")
        return 1
    print(f"\n{len(fresh)} scenarios: the committed vectors still describe "
          "the python reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
