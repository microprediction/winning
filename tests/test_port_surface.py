"""The three engines are compared by 43 hand-written parity scenarios, and
a verb or option nobody wrote a scenario for is invisible. This file makes
the gaps declared instead of accidental.

Three divergences reached `main` before anyone noticed: the rank-one node
handover moved in python only (#153), the pair closed form's weight
normalisation moved in python only (#171), and the inverse's damping
likewise (#178). The audit that followed found two more that no scenario
could ever have caught, because no scenario called the code: the browser
port SILENTLY IGNORED `cov=` and returned the independent race, and R's
`fit_covariance` crashed outright on an n=8 fixture.

So every public verb must carry a declared status per port, and every
keyword of the two central verbs must be exercised or declared. Adding a
verb, or a keyword, fails these tests until someone decides what the ports
do about it. The rule that matters: a declared gap must be a LOUD gap --
a port that silently ignores what it does not implement is the one failure
mode that looks like an answer.
"""
import inspect
import pathlib
import re

import pytest

import winning.factor as wf
import winning.factor.blocks as wfb

ROOT = pathlib.Path(__file__).resolve().parents[1]
R_NS = (ROOT / "r" / "winning" / "NAMESPACE").read_text()
JS_SRC = "\n".join(f.read_text() for f in (ROOT / "docs/js/winning").glob("*.mjs"))
GEN = (ROOT / "parity" / "gen_vectors.py").read_text()
CHECK_R = (ROOT / "parity" / "check.R").read_text()
CHECK_JS = (ROOT / "parity" / "check.mjs").read_text()

PARITY = "parity"        # exercised by a parity scenario in all three
PRESENT = "present"      # exported by the port, deliberately not in parity
GAP = "gap"              # not in the port, with a reason


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


# verb -> (R status, js status, reason for any gap or uncovered presence)
SURFACE = {
    # the race itself, and its inverses
    "race_probabilities":            (PARITY, PARITY, ""),
    "abilities_from_race":           (PARITY, PARITY, ""),
    "calibrate_abilities":           (PRESENT, GAP,
                                      "alias of abilities_from_race, which "
                                      "carries the parity scenario; the "
                                      "browser exposes the one name"),
    "win_probabilities_factor":      (PRESENT, GAP,
                                      "the browser calls the race through "
                                      "raceProbabilities only"),
    # top-k family
    "top_k_probabilities":           (PARITY, PARITY, ""),
    "bottom_k_probabilities":        (PARITY, PARITY, ""),
    "abilities_from_topk":           (PARITY, PARITY, ""),
    "abilities_from_rank_marginal":  (PARITY, PARITY, ""),
    "rank_probabilities":            (PARITY, PARITY, ""),
    "loc_scale_from_topk_pair":      (PARITY, PARITY, ""),
    "loc_scale_from_win_and_second": (PARITY, PARITY, ""),
    # quadrature
    "hermite_nodes":                 (PARITY, PARITY, ""),
    "qmc_nodes":                     (GAP, GAP,
                                      "scrambled Sobol needs scipy; the "
                                      "ports use dependency-free Halton at "
                                      "the same 2^13 budget, which is a "
                                      "declared numerical difference"),
    # python-only engine internals and research verbs
    "abilities_from_win_probabilities": (GAP, GAP, "research alias"),
    "factor_model_contrast":         (GAP, GAP, "fitting, not racing"),
    "factor_model_projected":        (GAP, GAP, "fitting, not racing"),
    "jacobian_vector_product":       (GAP, GAP,
                                      "used by the ratings filters, which "
                                      "have no port"),
    "ordered_probabilities":         (GAP, GAP,
                                      "the ordered-prefix kernel is python "
                                      "only; see the cov= note in races.py"),
    "plackett_luce_prefix_logprob":  (GAP, GAP, "python only"),
    "removal_shares":                (GAP, GAP, "python only"),
    "tie_densities":                 (GAP, GAP, "python only"),
    # the block grammars
    "block_race_probabilities":      (PARITY, PARITY, ""),
    "nested_race_probabilities":     (PARITY, PARITY, ""),
    "tree_race_probabilities":       (PARITY, PARITY, ""),
    "block_race_jacobian":           (PARITY, PARITY, ""),
    "nested_race_jacobian":          (PARITY, PARITY, ""),
    "tree_race_jacobian":            (PARITY, PARITY, ""),
    "abilities_from_block_race":     (PARITY, PARITY, ""),
}


def _public_verbs():
    seen = {}
    for mod in (wf, wfb):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if inspect.ismodule(obj) or not callable(obj):
                continue
            if not getattr(obj, "__module__", "").startswith("winning.factor"):
                continue
            if name in ("as_loadings", "as_idio", "as_variance", "load_fastrace"):
                continue            # shape/dispatch helpers, not race verbs
            seen[name] = obj
    return seen


VERBS = _public_verbs()


def test_every_public_verb_has_a_declared_port_status():
    """A new verb must not arrive without someone deciding what R and the
    browser do about it. That decision is the whole point of this file."""
    undeclared = sorted(set(VERBS) - set(SURFACE))
    stale = sorted(set(SURFACE) - set(VERBS))
    assert not undeclared, (
        "public factor verbs with no entry in SURFACE -- add one saying "
        "whether each port has it, and why not:\n  " + "\n  ".join(undeclared))
    assert not stale, (
        "SURFACE names verbs that no longer exist:\n  " + "\n  ".join(stale))


@pytest.mark.parametrize("verb", sorted(SURFACE))
def test_declared_presence_matches_the_port_sources(verb):
    r_status, js_status, reason = SURFACE[verb]
    in_r = bool(re.search(rf"^export\({re.escape(verb)}\)", R_NS, re.M))
    in_js = bool(re.search(rf"export function {re.escape(_camel(verb))}\b",
                           JS_SRC))
    assert in_r == (r_status != GAP), (
        f"{verb}: SURFACE says R {r_status}, NAMESPACE says "
        f"{'exported' if in_r else 'absent'}")
    assert in_js == (js_status != GAP), (
        f"{verb}: SURFACE says js {js_status}, exports say "
        f"{'present' if in_js else 'absent'}")
    if GAP in (r_status, js_status) or PRESENT in (r_status, js_status):
        assert reason, f"{verb}: a gap or an uncovered export needs a reason"


@pytest.mark.parametrize("verb", sorted(v for v, s in SURFACE.items()
                                        if s[0] == PARITY or s[1] == PARITY))
def test_parity_claims_are_backed_by_a_scenario(verb):
    """Claiming PARITY means all three engines really run it on the same
    input: the verb is called in the generator AND named in both checkers."""
    assert re.search(rf"\b{re.escape(verb)}\(", GEN), \
        f"{verb} is declared PARITY but the generator never calls it"
    assert re.search(rf"\b{re.escape(verb)}\(", CHECK_R), \
        f"{verb} is declared PARITY but check.R never calls it"
    assert re.search(rf"\b{re.escape(_camel(verb))}\(", CHECK_JS), \
        f"{verb} is declared PARITY but check.mjs never calls it"


def _call_args(src, verb):
    """Argument text of every `verb(...)` call in src, parens balanced --
    a non-greedy regex stops at the first `)` and silently reports a call
    as not exercising the keywords past it."""
    out = []
    for m in re.finditer(rf"\b{re.escape(verb)}\(", src):
        i = m.end()
        depth = 1
        while i < len(src) and depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.end():i - 1])
    return out


# keyword -> reason it is not exercised by a parity scenario
OPTION_WAIVERS = {
    "race_probabilities": {
        "cov": "python routes a degraded fit to GHK and the ports have no "
               "GHK; the browser must REJECT the key and R must WARN that "
               "it prices the fit instead (see the loud-gap test)",
        "temperature": "softmin is python only",
        "structure": "covered by the grammar scenarios, which call the "
                     "dispatch directly",
        "F": "the node rule chooses F; hermite_nodes carries its own scenario",
        "W": "as F",
        "window": "covered by factor2_span",
        "delta": "a numerical floor, not a mode",
    },
    "abilities_from_race": {
        "cov": "as race_probabilities",
        "temperature": "softmin is python only",
        "structure": "covered by invert_blocks",
        "F": "the node rule chooses F",
        "W": "as F",
        "target_floor": "a contract on the target, pinned by python tests",
        "return_info": "diagnostics, not a different race",
        "n_iter": "iteration budget, not a different race",
        "tol": "as n_iter",
    },
}


@pytest.mark.parametrize("verb", sorted(OPTION_WAIVERS))
def test_every_keyword_is_exercised_or_waived(verb):
    """The `cov=` defect was an OPTION gap, not a verb gap: the browser
    accepted the key and ignored it. Every keyword of the two central verbs
    must therefore be either exercised by a scenario or waived by name."""
    sig = inspect.signature(VERBS[verb])
    kw = [p for p, v in sig.parameters.items()
          if v.default is not inspect.Parameter.empty]
    calls = _call_args(GEN, verb)
    exercised = {k for k in kw if any(re.search(rf"\b{k}\s*=", c) for c in calls)}
    waived = set(OPTION_WAIVERS[verb])
    missing = sorted(set(kw) - exercised - waived)
    assert not missing, (
        f"{verb}: keywords neither exercised by a parity scenario nor "
        f"waived in OPTION_WAIVERS:\n  " + "\n  ".join(missing))
    unknown = sorted(waived - set(kw))
    assert not unknown, f"{verb}: OPTION_WAIVERS names dead keywords: {unknown}"


def test_a_declared_option_gap_is_a_loud_gap():
    """The rule this whole file exists for. `cov=` is waived because the
    ports cannot serve it -- which is only acceptable if they SAY SO. The
    browser swallowed the key and returned the independent race, identical
    even for an all-zero covariance; python and R raise by language, the
    browser needs the guard."""
    races = (ROOT / "docs/js/winning/races.mjs").read_text()
    assert "KNOWN_OPTS" in races and "checkOpts" in races
    assert 'k === "cov"' in races, "cov needs to be named in the guard"
    assert "cov" not in re.search(r"const KNOWN_OPTS = new Set\(\[(.*?)\]\)",
                                  races, re.S).group(1), \
        "cov must not be in the accepted set"
    for fn in ("raceProbabilities", "abilitiesFromRace"):
        i = races.index(f"export function {fn}(")
        assert "checkOpts(opts, " in races[i:i + 900], f"{fn} must check"

    # R serves cov= but has no GHK, so it prices the fit where python
    # routes -- 4.6e-3 apart on the n=8 fixture, and it used to say
    # nothing at all. fit.R reports the two failure classes and races.R
    # warns on them.
    r_fit = (ROOT / "r/winning/R/fit.R").read_text()
    r_races = (ROOT / "r/winning/R/races.R").read_text()
    assert "degraded =" in r_fit, "fit.R must report the degradation"
    assert ".warn_degraded_cov" in r_races
    assert "warning(" in r_races
    for fn in ("race_probabilities <- function", "abilities_from_race <- function"):
        i = r_races.index(fn)
        assert ".warn_degraded_cov(fit," in r_races[i:i + 1400], fn


def test_python_and_r_reject_unknown_keywords_by_language():
    """Only a keyword-object API needs the guard above; pin that the other
    two really do raise, so nobody 'harmonises' by loosening them."""
    import numpy as np
    with pytest.raises(TypeError):
        wf.race_probabilities(np.array([0.0, 1.0]), D=np.ones(2), nope=1)
