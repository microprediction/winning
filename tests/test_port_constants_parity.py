"""The three engines document one node rule; the numbers in it must
agree (#153: the rank-one Gauss-Hermite -> midpoint handover moved from
201 to 80 in python only, and the ports kept 201 while all three
claimed to match)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

PY = ROOT / "winning" / "factor" / "races.py"
R = ROOT / "r" / "winning" / "R" / "races.R"
JS = ROOT / "docs" / "js" / "winning" / "races.mjs"


def _handover(text, pattern):
    m = re.search(pattern, text)
    assert m, pattern
    return int(m.group(1))


def test_rank_one_handover_threshold_agrees_across_ports():
    py = _handover(PY.read_text(), r"r == 1 and np\.ceil\(8\.0 \* sharp\) > (\d+)")
    r = _handover(R.read_text(), r"r == 1 && ceiling\(8 \* sharp\) > (\d+)")
    js = _handover(JS.read_text(), r"r === 1 && Math\.ceil\(8 \* sharp\) > (\d+)")
    assert py == r == js, (py, r, js)


def test_ports_carry_the_inverse_safeguards():
    """The scale-aware warm start, the pair closed form and the adaptive
    damping (#163, #149) are in all three engines."""
    for path, markers in [
        (R, ["scale <- sqrt(median(Dn)", "qnorm(target[1])", "2 / (2 - lam)"]),
        (JS, ["Math.sqrt(med(Dn)", "invNormalRational(target[0])", "2 / (2 - lam)"]),
    ]:
        text = path.read_text()
        for marker in markers:
            assert marker in text, (path.name, marker)
