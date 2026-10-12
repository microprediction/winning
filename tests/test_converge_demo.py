"""docs/converge.html keeps its expensive reference off the main thread.

#647: the default rank-2 Factor fixture takes the 8,192-node QMC rule
(correctly; #534), and the page used to compute it synchronously before
its first render, freezing the tab for ~48 s. The reference now runs in
docs/js/converge_worker.mjs.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "converge.html").read_text()
SCRIPT = PAGE[PAGE.index('<script type="module">'):PAGE.rindex("</script>")]


def _body(name):
    start = SCRIPT.index(f"function {name}(")
    nxt = re.search(r"\n    function \w+\(", SCRIPT[start + 1:])
    return SCRIPT[start:start + 1 + nxt.start()] if nxt else SCRIPT[start:]


def test_reference_is_computed_in_a_module_worker():
    assert (ROOT / "docs" / "js" / "converge_worker.mjs").exists()
    assert 'new Worker(new URL("./js/converge_worker.mjs"' in SCRIPT
    assert '{ type: "module" }' in SCRIPT


def test_run_does_not_price_the_structured_reference_on_the_main_thread():
    run = _body("run")
    # the only main-thread race in run() is the cheap independent one
    assert run.count("raceProbabilities(") == 1
    assert "D: C.map(" in run
    assert "fitGrammar(" not in run and "latticeReference(" not in run

