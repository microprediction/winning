"""The two pooled ranking rows of the by-source table: GSS (two items) and Jester (three files).

Each item or file is calibrated and scored on its own and pooled at the observation level,
with the refit respondent bootstrap of sensitivity.py run on the group. Uses
sensitivity.py's own run_grouped and bootstrap_refit_grouped, so nothing here is new
methodology; it only names the two groups that sensitivity.py does not.

Usage:  python bysource_pooled.py [reps] [GSS|Jester]     (default 2000, both)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensitivity as s
from heldout_score import load_all

GROUPS = {"GSS": ["GSS socialization", "GSS job values"],
          "Jester": ["Jester file 1", "Jester file 2", "Jester file 3"]}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    which = sys.argv[2:] or list(GROUPS)
    data = load_all()
    for name in which:
        Rs = [data[k] for k in GROUPS[name]]
        g = s.run_grouped(Rs)
        lo, hi, k, kind = s.bootstrap_refit_grouped(Rs, reps=reps)
        print(f"{name:<8} pooled gain {g:+.4f}  refit interval [{lo:+.4f}, {hi:+.4f}]  {k} reps  {kind}",
              flush=True)


if __name__ == "__main__":
    main()
