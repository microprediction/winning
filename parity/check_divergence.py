"""Behavioural parity across the ports: which inputs each one ACCEPTS.

`check_vectors.py`, `check.mjs`, `check.R` and `check.jl` compare the
ports' VALUES on well-formed inputs. Nothing compared their DECISIONS
on malformed ones, and that is where they drifted: a port that quietly
recycles a short `D`, or returns NaN where another raises, agrees with
everyone on every vector in those files and still prices a different
race.

Every divergence this found was real, and none had been reported:

  mu_nan / mu_inf     python and browser return NaN; R and julia refuse
  D_too_short/long    R recycles: D=c(2,9) equals D=c(2,9,2,9) exactly
  inv_p_too_short     R answers a 2-runner race from a 4-entry D
  V_scalar            R and julia refuse the documented scalar spelling
  D_zero              julia returns NaN; the others refuse
  mu_empty            the browser answers an empty field

Run: python parity/check_divergence.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = os.path.join(HERE, "divergence_cases.json")

RUNNERS = {
    "python": [sys.executable, os.path.join(HERE, "divergence_py.py"), CASES],
    "browser": ["node", os.path.join(HERE, "divergence_js.mjs"), CASES],
    "R": ["Rscript", os.path.join(HERE, "divergence_r.R"), CASES],
    "julia": ["julia", "--color=no", os.path.join(HERE, "divergence_jl.jl"),
              CASES],
}

# Divergences that are deliberate. Each needs a reason, not just an id:
# an entry here is a claim that the ports SHOULD differ.
EXPECTED = {}


def _verdict(v):
    if isinstance(v, dict):
        v = v.get("verdict", "?")
    if isinstance(v, list):          # R's toJSON wraps scalars
        v = v[0] if v else "?"
    return str(v)


def main():
    results, missing = {}, []
    for name, cmd in RUNNERS.items():
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                 text=True, timeout=900)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            missing.append(name)
            continue
        if out.returncode != 0 or not out.stdout.strip():
            missing.append(name)
            continue
        results[name] = json.loads(out.stdout.strip().splitlines()[-1])
    if len(results) < 2:
        print(f"skip: need two ports, have {sorted(results)} "
              f"(missing {sorted(missing)})")
        return 0
    if missing:
        print(f"note: not run for {sorted(missing)}")

    ids = list(json.load(open(CASES))["cases"])
    diverged = []
    for case in ids:
        cid = case["id"]
        got = {p: _verdict(r.get(cid, "?")) for p, r in results.items()}
        if len(set(got.values())) > 1 and cid not in EXPECTED:
            diverged.append((cid, got))

    width = max(len(c["id"]) for c in ids)
    for case in ids:
        cid = case["id"]
        got = {p: _verdict(r.get(cid, "?")) for p, r in results.items()}
        flag = "  <== DIVERGES" if any(cid == d[0] for d in diverged) else ""
        cols = "  ".join(f"{p}={got[p]}" for p in sorted(got))
        print(f"  {cid:{width}s}  {cols}{flag}")

    if diverged:
        print(f"\n{len(diverged)} of {len(ids)} cases diverge across ports")
        print("A port that accepts what another refuses prices a different "
              "race on the same input. Fix it, or record it in EXPECTED "
              "with the reason.")
        return 1
    print(f"\nall {len(ids)} cases agree across {sorted(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
