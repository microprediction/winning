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

# Divergences that are known. Each needs a reason, not just an id. The
# list is STRICT in both directions: a case here that diverges is
# reported and tolerated, and a case here that has started to AGREE
# fails, so the list cannot outlive what it waives.
# `ports` names the ports whose DISAGREEMENT the waiver is about. A
# waiver can only be judged stale when all of them actually ran: with
# julia absent, python and the browser agreeing about a fractional
# depth says nothing about whether julia still refuses it, and calling
# the waiver stale there told the reader to delete a live one (#310).
EXPECTED = {
    "inv_p_huge": {
        "why": "#326 -- python rescues a target whose entries are finite "
               "but whose SUM overflows, by dividing by the max before "
               "normalising, and recovers the SAME abilities as the "
               "finite target with identical ratios. That is #300's fix, "
               "python-only so far; the other three still refuse. Goes "
               "when the rescue is ported.",
        "ports": ["python", "R", "browser", "julia"],
    },
    "D_tiny": {
        "why": "#304 -- every port prices a contestant whose sd the "
               "lattice cannot resolve at zero, and they disagree by "
               "1.5e-4 about the others. The disagreement is a symptom; "
               "the zero is the defect.",
        "ports": ["python", "R", "browser", "julia"],
    },
}


# Two ports can agree to ACCEPT and still answer differently. That is
# the quieter half of the same defect, so compare the numbers too --
# loosely. check_vectors.py and its siblings own tight value parity on
# well-formed inputs; this threshold is here to catch a port answering
# a DIFFERENT QUESTION, not to police the last ulp.
VALUE_TOL = 1e-9

# ...except where the algorithm's OWN stopping rule is looser than that.
# Both inverses solve to a residual of 1e-8 in probability space, which
# leaves abilities free to differ by ~2e-9 between ports while every one
# of them is converged and correct. Asserting agreement tighter than the
# tolerance the code promises tests the arithmetic, not the port.
VERB_TOL = {"inverse": 1e-6}


def _value(v):
    if not isinstance(v, dict):
        return None
    got = v.get("value")
    if got is None:
        return None
    if not isinstance(got, list):
        got = [got]
    try:
        return [float(x[0] if isinstance(x, list) else x) for x in got]
    except (TypeError, ValueError):
        return None


def _value_gap(vals):
    """The worst disagreement between ports, or None if not comparable."""
    named = [(p, v) for p, v in vals.items() if v]
    if len(named) < 2:
        return None
    widths = {len(v) for _, v in named}
    if len(widths) > 1:
        return float("inf")          # different shapes IS a disagreement
    ref = named[0][1]
    gap = 0.0
    for _, v in named[1:]:
        for a, b in zip(ref, v):
            gap = max(gap, abs(a - b))
    return gap


def _verdict(v):
    if isinstance(v, dict):
        v = v.get("verdict", "?")
    if isinstance(v, list):          # R's toJSON wraps scalars
        v = v[0] if v else "?"
    return str(v)


def main():
    # An ABSENT toolchain is a skip; a toolchain that is present and
    # whose runner FAILS is a failure. Collapsing the two let a port
    # drop silently out of the comparison and the scan still print
    # "all cases agree" -- the one report this tool must never make
    # when it has not actually compared the port.
    results, absent, broken = {}, [], {}
    for name, cmd in RUNNERS.items():
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                 text=True, timeout=900)
        except FileNotFoundError:
            absent.append(name)
            continue
        except subprocess.TimeoutExpired:
            broken[name] = "timed out after 900s"
            continue
        if out.returncode != 0:
            broken[name] = (f"exit {out.returncode}: "
                            + (out.stderr.strip().splitlines() or [""])[-1])
            continue
        if not out.stdout.strip():
            broken[name] = "produced no output"
            continue
        try:
            results[name] = json.loads(out.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError as exc:
            broken[name] = f"unreadable output: {exc}"
    if broken:
        for name, why in sorted(broken.items()):
            print(f"FAILED to run the {name} port: {why}")
        print("The scan cannot speak for a port it could not run.")
        return 1
    if len(results) < 2:
        print(f"skip: need two ports, have {sorted(results)} "
              f"(absent {sorted(absent)})")
        return 0
    if absent:
        print(f"note: toolchain absent, not compared: {sorted(absent)}")

    ids = list(json.load(open(CASES))["cases"])
    # A waiver is only ever examined while walking the cases, so one
    # whose case has been DELETED is never looked at again and sits
    # there forever. Same failure as a stale waiver, one step earlier.
    orphans = sorted(set(EXPECTED) - {c["id"] for c in ids})
    if orphans:
        for cid in orphans:
            print(f"EXPECTED case {cid!r} is not in the case file: "
                  f"{EXPECTED[cid]['why']}")
        print("A waiver for a case nobody runs waives nothing.")
        return 1
    diverged, value_gaps, stale, unjudged = [], {}, [], {}
    for case in ids:
        cid = case["id"]
        got = {p: _verdict(r.get(cid, "?")) for p, r in results.items()}
        tol = VERB_TOL.get(case["verb"], VALUE_TOL)
        gap = None
        if set(got.values()) == {"ACCEPT"}:
            gap = _value_gap({p: _value(r.get(cid)) for p, r in
                              results.items()})
        # a case diverges if the ports DECIDE differently or ANSWER
        # differently; an EXPECTED entry has to clear both to be stale
        differs = (len(set(got.values())) > 1
                   or (gap is not None and gap > tol))
        if cid in EXPECTED:
            need = set(EXPECTED[cid]["ports"])
            absent_here = need - set(results)
            if absent_here:
                unjudged[cid] = sorted(absent_here)
            elif not differs:
                stale.append(cid)
            continue
        if len(set(got.values())) > 1:
            diverged.append((cid, got))
        elif gap is not None and gap > tol:
            value_gaps[cid] = gap

    width = max(len(c["id"]) for c in ids)
    for case in ids:
        cid = case["id"]
        got = {p: _verdict(r.get(cid, "?")) for p, r in results.items()}
        if cid in unjudged:
            flag = ("  <== known, not judged: "
                    + ", ".join(unjudged[cid]) + " absent")
        elif cid in EXPECTED:
            flag = "  <== known, see EXPECTED"
        else:
            flag = "  <== DIVERGES" if any(cid == d[0] for d in diverged) else (
                f"  <== SAME VERDICT, DIFFERENT ANSWER "
                f"({value_gaps[cid]:.2e})" if cid in value_gaps else "")
        cols = "  ".join(f"{p}={got[p]}" for p in sorted(got))
        print(f"  {cid:{width}s}  {cols}{flag}")

    if stale:
        print()
        for cid in stale:
            print(f"EXPECTED case {cid!r} no longer diverges: "
                  f"{EXPECTED[cid]['why']}")
        print("Remove it from EXPECTED -- a waiver that outlives the "
              "defect hides the next one.")
        return 1
    if value_gaps and not diverged:
        print(f"\n{len(value_gaps)} of {len(ids)} cases are ACCEPTED by "
              "every port with different answers")
        print("Agreeing to accept is not agreeing on the race.")
        return 1
    if diverged:
        print(f"\n{len(diverged)} of {len(ids)} cases diverge across ports")
        print("A port that accepts what another refuses prices a different "
              "race on the same input. Fix it, or record it in EXPECTED "
              "with the reason.")
        return 1
    if value_gaps:
        print(f"\n{len(value_gaps)} value gaps alongside the divergences")
    print(f"\nall {len(ids)} cases agree, in verdict and answer, across {sorted(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
