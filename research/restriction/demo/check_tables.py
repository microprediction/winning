"""Check every numeric figure in the paper against the committed results files.

The mathematical claims are checked in JavaScript, by `demo/run_checks.js`, which needs
no data. This script covers the other half: every number the paper quotes from a run has
to appear in the output file that produced it.

The failure mode it exists to catch is the one two reviewers found by hand. A number is
edited in one place and not the other, or a gain from one analysis population is paired
with a tail from a different one, and nothing in the build complains.

Usage:  python check_tables.py           from research/restriction
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT.parent.parent / "papers" / "thurstone_humans" / "paper.tex"
RESULTS = ROOT / "results"

# Each entry: the figure as printed in the paper, the results file that must contain it,
# and what it is. A figure is matched if it appears in the file at the same rounding.
CLAIMS = [
    # ---- Yeon and Rahnev, results/yeonrahnev.txt
    ("0.4942", "yeonrahnev.txt", "exp 1 renormalization loss"),
    ("0.4795", "yeonrahnev.txt", "exp 1 race loss"),
    ("0.0147", "yeonrahnev.txt", "exp 1 gain"),
    ("0.0118", "yeonrahnev.txt", "exp 1 bootstrap lower"),
    ("0.0179", "yeonrahnev.txt", "exp 1 bootstrap upper"),
    ("0.6276", "yeonrahnev.txt", "exp 2 renormalization loss"),
    ("0.5999", "yeonrahnev.txt", "exp 2 race loss"),
    ("0.0278", "yeonrahnev.txt", "exp 2 gain"),
    ("0.0234", "yeonrahnev.txt", "exp 2 bootstrap lower"),
    ("0.0324", "yeonrahnev.txt", "exp 2 bootstrap upper"),
    ("0.3967", "yeonrahnev.txt", "advance-warning renormalization loss"),
    ("0.3928", "yeonrahnev.txt", "advance-warning race loss"),
    ("0.0038", "yeonrahnev.txt", "advance-warning gain"),
    ("0.7802", "yeonrahnev.txt", "observed accuracy, menu after"),
    ("0.8504", "yeonrahnev.txt", "observed accuracy, menu before"),
    ("0.8493", "yeonrahnev.txt", "renormalization predicted accuracy"),
    ("0.8347", "yeonrahnev.txt", "race predicted accuracy"),
    # ---- Getty, results/getty.txt
    ("0.8144", "getty.txt", "all rows renormalization loss"),
    ("0.7872", "getty.txt", "all rows race loss"),
    ("0.0272", "getty.txt", "all rows gain"),
    ("0.4504", "getty.txt", "signal rows renormalization loss"),
    ("0.4393", "getty.txt", "signal rows race loss"),
    ("0.0111", "getty.txt", "signal rows gain"),
    ("0.0021", "getty.txt", "signal rows bootstrap lower"),
    ("0.0225", "getty.txt", "signal rows bootstrap upper"),
    ("0.0453", "getty.txt", "condition 1 gain"),
    ("0.0127", "getty.txt", "condition 2 gain"),
    ("0.0490", "getty.txt", "condition 3 gain"),
    ("0.103", "getty.txt", "condition 1 within-set confusion"),
    ("0.790", "getty.txt", "condition 2 within-set confusion"),
    ("0.335", "getty.txt", "condition 3 within-set confusion"),
    # ---- tones, results/tones.txt
    ("1.1176", "tones.txt", "narrow ten to six renormalization loss"),
    ("1.1309", "tones.txt", "narrow ten to six race loss"),
    ("0.0133", "tones.txt", "narrow ten to six gain"),
    ("0.0051", "tones.txt", "narrow ten to eight gain"),
    ("0.0173", "tones.txt", "wide ten to six gain"),
    ("0.0057", "tones.txt", "wide ten to eight gain"),
    # ---- Rouder lines, results/rouder_chunk.txt
    ("0.7846", "rouder_chunk.txt", "all blocks renormalization loss"),
    ("0.7980", "rouder_chunk.txt", "all blocks race loss"),
    ("0.0134", "rouder_chunk.txt", "all blocks gain"),
    ("0.0176", "rouder_chunk.txt", "all blocks bootstrap lower"),
    ("0.0104", "rouder_chunk.txt", "all blocks bootstrap upper"),
    ("0.0322", "rouder_chunk.txt", "twelve to two gain"),
    ("0.0090", "rouder_chunk.txt", "twelve to four gain"),
    # ---- recognition foils, results/recognition.txt
    ("0.3859", "recognition.txt", "all foils renormalization loss"),
    ("0.3821", "recognition.txt", "all foils race loss"),
    ("0.0037", "recognition.txt", "all foils gain"),
    ("0.0029", "recognition.txt", "all foils bootstrap lower"),
    ("0.0047", "recognition.txt", "all foils bootstrap upper"),
    # ---- Wills categories, results/wills.txt
    ("0.6839", "wills.txt", "renormalization loss"),
    ("0.6540", "wills.txt", "race loss"),
    ("0.0299", "wills.txt", "gain"),
    ("0.0175", "wills.txt", "bootstrap lower"),
    ("0.0717", "wills.txt", "bootstrap upper"),
    ("0.0413", "wills.txt", "disallowed category 2 gain"),
    ("0.0485", "wills.txt", "disallowed category 3 gain"),
    # ---- news slates and the observed-restriction table
    ("0.0496", "mind_slates.txt", "news slate gain"),
    # ---- gain by size of the surviving menu
    ("0.0412", "gain_by_size.txt", "sushi pairwise gain"),
    ("0.0263", "gain_by_size.txt", "sushi |T|=3 gain"),
    ("0.0130", "gain_by_size.txt", "GSS socialization pairwise gain"),
]

# Figures the paper prints that are computed in the paper itself rather than by a script.
# Each is verified by the JavaScript suite instead; listed so the inventory is complete.
CHECKED_IN_JS = {
    "0.9078": "concentrated shares, favourite when the tail is withdrawn",
    "0.9091": "concentrated shares, renormalized favourite",
    "0.8028": "concentrated shares, survivor when the leader is withdrawn",
    "0.9000": "concentrated shares, renormalized survivor",
}


def read(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return path.read_text()


def paper_table_figures(paper):
    """Every decimal figure printed inside a table environment."""
    out = []
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", paper, re.S):
        label = re.search(r"\\label\{(tab:[^}]+)\}", block)
        name = label.group(1) if label else "unlabelled"
        for m in re.findall(r"[+-]?\d*\.\d{3,4}", block):
            out.append((name, m.lstrip("+")))
    return out


def main():
    paper = PAPER.read_text() if PAPER.exists() else ""
    if not paper:
        print(f"cannot read {PAPER}")
        return 1

    # ---- hard check: every figure the paper takes from a run is in that run's output
    missing_file, not_in_results = [], []
    for figure, fname, what in CLAIMS:
        text = read(fname)
        if text is None:
            missing_file.append((fname, what))
            continue
        bare = figure.lstrip("0") if figure.startswith("0.") else figure
        if figure not in text and bare not in text:
            not_in_results.append((figure, fname, what))

    print(f"{len(CLAIMS)} quoted figures traced to {len({c[1] for c in CLAIMS})} results files")
    print(f"{len(CHECKED_IN_JS)} further figures are verified by demo/run_checks.js")

    bad = 0
    for fname, what in missing_file:
        print(f"  MISSING RESULTS FILE  {fname}  ({what})")
        bad += 1
    for figure, fname, what in not_in_results:
        print(f"  NOT IN {fname:<22} {figure}  ({what})")
        bad += 1
    if not bad:
        print("  all traced")

    # ---- audit: table figures with no matching figure in any results file
    all_results = "\n".join(
        (RESULTS / f).read_text() for f in sorted(p.name for p in RESULTS.glob("*.txt")))
    # every decimal figure in every results file, as a float, so the paper's rounding
    # can be matched rather than its exact spelling
    pool = set()
    for m in re.findall(r"[+-]?\d*\.\d+", all_results):
        try:
            pool.add(float(m))
        except ValueError:
            pass

    def sourced(fig):
        try:
            v = float(fig)
        except ValueError:
            return False
        dp = len(fig.split(".")[1])
        return any(abs(round(w, dp) - v) < 1e-12 for w in pool)

    unsourced = {}
    for name, fig in paper_table_figures(paper):
        if sourced(fig):
            continue
        bare = fig.lstrip("0") if fig.startswith("0.") else fig
        if fig in CHECKED_IN_JS or bare in CHECKED_IN_JS:
            continue
        unsourced.setdefault(name, set()).add(fig)

    # ---- arithmetic check on the derived excess column: excess = gain - null median
    print("\nderived columns, checked by arithmetic rather than traced to a file:")
    for label, gain, med, printed in [
            ("Sushi", 0.0111, -0.0062, 0.0174),
            ("menu experiment, forced choice", 0.0265, -0.0041, 0.0306),
            ("menu experiment, all subjects", 0.0100, -0.0023, 0.0123)]:
        got = round(gain - med, 4)
        mark = "ok" if abs(got - printed) <= 1e-4 else "MISMATCH"
        note = " (rounding of the underlying values)" if got != printed and mark == "ok" else ""
        print(f"  {label:<32} {gain:+.4f} - ({med:+.4f}) = {got:+.4f} against {printed:+.4f}  {mark}{note}")

    print("\nfigures with no committed run behind them:")
    print("  tab:null, +0.0265, the pooled forced-choice gain. menus_heldout.txt has")
    print("  +0.0442 and +0.0140 for the two experiments' forced-choice subgroups and")
    print("  +0.0100 for all subjects pooled, but no pooled forced-choice figure.")

    total = len(paper_table_figures(paper))
    n_unsourced = sum(len(v) for v in unsourced.values())
    print(f"\n{total} figures appear in tables; {n_unsourced} have no match in results/*.txt")
    for name in sorted(unsourced):
        print(f"  {name}: {', '.join(sorted(unsourced[name]))}")
    print("\nUnmatched figures are not necessarily wrong. Derived quantities, ratios and")
    print("digitized inputs are computed in the paper or live outside results/. Each one")
    print("should have a traceable origin, and this list is what an audit reads.")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
