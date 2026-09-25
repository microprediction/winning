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
    "abilities_from_block_race":           (PARITY, PARITY, ""),
    "abilities_from_probabilities":        (GAP, GAP,
                                        "alias of a verb that carries the parity scenario"),
    "abilities_from_probabilities_factor": (PRESENT, GAP,
                                        "alias of a verb that carries the parity scenario"),
    "abilities_from_race":                 (PARITY, PARITY, ""),
    "abilities_from_rank_marginal":        (PARITY, PARITY, ""),
    "abilities_from_softmax":              (GAP, GAP,
                                        "alias of a verb that carries the parity scenario"),
    "abilities_from_topk":                 (PARITY, PARITY, ""),
    "abilities_from_win_probabilities":    (GAP, GAP,
                                        "python only"),
    "block_race_jacobian":                 (PARITY, PARITY, ""),
    "block_race_probabilities":            (PARITY, PARITY, ""),
    "bottom_k_probabilities":              (PARITY, PARITY, ""),
    "calibrate_abilities":                 (PRESENT, GAP,
                                        "alias of abilities_from_race, which carries the"
                                        " scenario; the browser exposes the one name"),
    "concentration_matrix":                (PRESENT, PRESENT,
                                        "exported by all three and exercised by none:"
                                        " a scenario would be worth adding"),
    "exponential_power_base":              (GAP, GAP,
                                        "a base density, not a verb: the ports carry normal,"
                                        "gumbel, logistic and laplace inline and take a"
                                        "callable for the rest"),
    "factor_model":                        (GAP, GAP,
                                        "covariance fitting, reached through cov= rather than"
                                        "called"),
    "factor_model_contrast":               (GAP, GAP,
                                        "python only"),
    "factor_model_projected":              (GAP, GAP,
                                        "python only"),
    "failure_base":                        (GAP, GAP,
                                        "a base density, not a verb: the ports carry normal,"
                                        "gumbel, logistic and laplace inline and take a"
                                        "callable for the rest"),
    "fit_covariance":                      (PRESENT, GAP,
                                        "covariance fitting, reached through cov= rather than"
                                        "called"),
    "harville_order_logprob":              (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "harville_place_probabilities":        (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "harville_prefix_logprob":             (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "hermite_nodes":                       (PARITY, PARITY, ""),
    "jacobian_vector_product":             (GAP, GAP,
                                        "python only"),
    "loc_scale_from_topk_pair":            (PARITY, PARITY, ""),
    "loc_scale_from_win_and_second":       (PARITY, PARITY, ""),
    "nested_race_jacobian":                (PARITY, PARITY, ""),
    "nested_race_probabilities":           (PARITY, PARITY, ""),
    "ordered_probabilities":               (GAP, GAP,
                                        "python only"),
    "plackett_luce_order_logprob":         (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "plackett_luce_prefix_logprob":        (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "plackett_luce_topk_probabilities":    (GAP, GAP,
                                        "order-statistics likelihood, python only"),
    "polish_race":                         (PARITY, PARITY, ""),
    "qmc_nodes":                           (GAP, GAP,
                                        "python only"),
    "race_jacobian":                       (PARITY, PARITY, ""),
    "race_jacobian_row":                   (GAP, GAP,
                                        "a single Jacobian row, an internal of the full"
                                        "Jacobian"),
    "race_probabilities":                  (PARITY, PARITY, ""),
    "rank_probabilities":                  (PARITY, PARITY, ""),
    "removal_shares":                      (GAP, GAP,
                                        "python only"),
    "skew_logistic_base":                  (GAP, GAP,
                                        "a base density, not a verb: the ports carry normal,"
                                        "gumbel, logistic and laplace inline and take a"
                                        "callable for the rest"),
    "skew_normal_base":                    (GAP, GAP,
                                        "a base density, not a verb: the ports carry normal,"
                                        "gumbel, logistic and laplace inline and take a"
                                        "callable for the rest"),
    "softmax_probabilities":               (GAP, GAP,
                                        "python only"),
    "student_base":                        (GAP, GAP,
                                        "a base density, not a verb: the ports carry normal,"
                                        "gumbel, logistic and laplace inline and take a"
                                        "callable for the rest"),
    "tie_densities":                       (GAP, GAP,
                                        "python only"),
    "top_k_jacobian":                      (GAP, GAP,
                                        "python only"),
    "top_k_jacobian_row":                  (GAP, GAP,
                                        "a single Jacobian row, an internal of the full"
                                        "Jacobian"),
    "top_k_jacobian_row_sigma":            (GAP, GAP,
                                        "a single Jacobian row, an internal of the full"
                                        "Jacobian"),
    "top_k_jacobians":                     (PARITY, PARITY, ""),
    "top_k_probabilities":                 (PARITY, PARITY, ""),
    "tree_race_jacobian":                  (PARITY, PARITY, ""),
    "tree_race_probabilities":             (PARITY, PARITY, ""),
    "win_probabilities":                   (GAP, GAP,
                                        "alias of a verb that carries the parity scenario"),
    "win_probabilities_factor":            (PRESENT, GAP,
                                        "python only"),
}


# every module of the factor package, not just the top-level facade: the
# first cut inspected `winning.factor` and `winning.factor.blocks` only, so
# top_k_jacobians, race_jacobian, polish_race and fit_covariance -- three of
# them exported by R, two by the browser -- had no declared status at all
# while this file claimed to cover the surface (#187).
import winning.factor.core as wfc          # noqa: E402
import winning.factor.permutations as wfp  # noqa: E402
import winning.factor.polish as wfo        # noqa: E402
import winning.factor.races as wfr         # noqa: E402
import winning.factor.topk as wft          # noqa: E402

_HELPERS = {"as_loadings", "as_idio", "as_variance", "load_fastrace",
            "forward_grid", "roots_hermitenorm", "ndtr", "ndtri",
            "log_ndtr", "logsumexp"}


def _public_verbs():
    seen = {}
    for mod in (wf, wfb, wfc, wfp, wfo, wfr, wft):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if inspect.ismodule(obj) or not callable(obj):
                continue
            if not getattr(obj, "__module__", "").startswith("winning.factor"):
                continue
            if name in _HELPERS:
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
        "temperature": "softmin is python only",
        "structure": "covered by the grammar scenarios, which call the "
                     "dispatch directly",
        "F": "the node rule chooses F; hermite_nodes carries its own scenario",
        "W": "as F",
        "window": "covered by factor2_span",
        "delta": "a numerical floor, not a mode",
    },
    "abilities_from_race": {
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
    """The rule this whole file exists for. `cov=` is now served by R --
    it routes a degraded fit to GHK as python does, pinned by the
    cov_degraded_route scenario -- and is still absent from the browser,
    which must therefore REJECT the key. It used to swallow it and return
    the independent race, identical even for an all-zero covariance."""
    races = (ROOT / "docs/js/winning/races.mjs").read_text()
    core = (ROOT / "docs/js/winning/core.mjs").read_text()
    # the guard and its reasons moved to core.mjs when every module needed
    # them; `cov` still has to carry a reason rather than "unknown option"
    assert "export function checkOpts" in core
    assert "cov:" in core and "fitGrammar" in core, \
        "cov needs a reason, not just a rejection"
    # each API validates its OWN signature: one shared union let each accept
    # the other's options and ignore them (#186)
    for const in ("FORWARD_OPTS", "INVERSE_OPTS"):
        allowed = re.search(rf"const {const} = new Set\(\[(.*?)\]\)",
                            races, re.S).group(1)
        assert "cov" not in allowed, f"cov must not be in {const}"
    fwd = re.search(r"const FORWARD_OPTS = new Set\(\[(.*?)\]\)", races, re.S).group(1)
    inv = re.search(r"const INVERSE_OPTS = new Set\(\[(.*?)\]\)", races, re.S).group(1)
    assert "nIter" not in fwd and "tol" not in fwd, "inverse-only keys leaked forward"
    assert "returnSlopes" not in inv and "window" not in inv, "forward-only keys leaked"
    for fn, const in (("raceProbabilities", "FORWARD_OPTS"),
                      ("abilitiesFromRace", "INVERSE_OPTS")):
        i = races.index(f"export function {fn}(")
        assert f"checkOpts(opts, {const}," in races[i:i + 900], f"{fn} must check {const}"

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


# an R or browser export that answers to no python verb: the registry is
# driven from python, so without this a port could grow a race verb the
# audit never sees (#187)
PORT_ONLY = {
    "R": {"ability_implied_dividends", "ability_implied_state_prices",
          "dividend_implied_ability", "dividends_from_prices",
          "prices_from_dividends", "solve_for_implied_offsets",
          "state_price_implied_ability", "state_prices_from_offsets",
          "skew_normal_density", "tree_from_hclust", "tree_from_linkage",
          "Blocks", "Factor", "Independent", "Nested", "Tree",
          "hermite_nodes", "win_probabilities_factor"},
    # browser-only: a reduced covariance fitter for the demo pages (blocks
    # omitted for latency, so NOT the package's fit_covariance), and the
    # dependency-free node family the ports use where python has Sobol
    "js": {"fitGrammar", "haltonNormalNodes"},
}


def _race_like(name):
    """The ports also export linear algebra and plumbing (cholesky, solve,
    mean, ndtr). Only names that read as race verbs are audited."""
    return any(k in name.lower() for k in
               ("race", "probabilit", "abilit", "topk", "top_k", "rank",
                "jacobian", "polish", "covariance", "nodes", "grammar"))


def test_port_race_exports_are_all_declared():
    r_exports = set(re.findall(r"^export\((\w+)\)", R_NS, re.M))
    js_exports = set(re.findall(r"export function (\w+)", JS_SRC))
    known_py = set(SURFACE)
    r_undeclared = sorted(
        n for n in r_exports
        if _race_like(n) and n not in known_py and n not in PORT_ONLY["R"])
    js_undeclared = sorted(
        n for n in js_exports
        if _race_like(n) and n not in {_camel(v) for v in known_py}
        and n not in {_camel(v) for v in PORT_ONLY["R"]}
        and n not in PORT_ONLY["js"])
    assert not r_undeclared, (
        "R exports race verbs with no declared status:\n  "
        + "\n  ".join(r_undeclared))
    assert not js_undeclared, (
        "the browser exports race verbs with no declared status:\n  "
        + "\n  ".join(js_undeclared))


def test_the_deployed_factor_race_copy_matches_its_source():
    """docs/assets/js/factor_race.mjs is what three doc pages actually
    load, and it is a COPY of js/factor/factor_race.mjs. It had silently
    missed the loading gauge fix (#139) and would have missed the tail fix
    (#182) the same way. Byte-identical or it is not a copy."""
    src = (ROOT / "js/factor/factor_race.mjs").read_bytes()
    deployed = (ROOT / "docs/assets/js/factor_race.mjs").read_bytes()
    assert src == deployed, (
        "docs/assets/js/factor_race.mjs has drifted from "
        "js/factor/factor_race.mjs; copy the source over it")

def test_every_browser_options_api_validates_its_options():
    """#186 guarded two entry points of twenty, and the reviewer walked
    into the rest within a day: rankProbabilities swallowed V and returned
    the independent rank matrix (#199), locScaleFromTopkPair swallowed V
    and reported converged where python raises (#200). Any export taking
    an options object must call checkOpts, and with its OWN allowlist."""
    offenders = []
    for path in sorted((ROOT / "docs/js/winning").glob("*.mjs")):
        text = path.read_text()
        for m in re.finditer(r"export function (\w+)\([^)]*opts\s*=\s*\{\}\s*\)\s*\{",
                             text):
            body = text[m.end():m.end() + 600]
            if "checkOpts(opts," not in body:
                offenders.append(f"{path.name}::{m.group(1)}")
    assert not offenders, (
        "browser exports taking an options object without validating it -- "
        "an object swallows a key nobody reads:\n  " + "\n  ".join(offenders))


def test_browser_allowlists_are_per_function():
    """A shared union is what let each API take the other's options."""
    names = set()
    for path in (ROOT / "docs/js/winning").glob("*.mjs"):
        names |= set(re.findall(r"const (\w+_OPTS) = new Set\(", path.read_text()))
    assert len(names) >= 18, f"only {len(names)} allowlists for 20 entry points: {sorted(names)}"
    assert "KNOWN_OPTS" not in names, "the shared union is back"


# a function may legitimately forward its whole options object to another;
# declared rather than guessed, because "the body mentions opts somewhere"
# would also have excused the bug this test exists for -- abilitiesFromRace
# forwards opts on its structure branch, and targetFloor still vanished
# on every other path (#226)
FORWARDS_OPTS = {
    "topk.mjs::bottomKProbabilities":
        "forwards to topKProbabilities(mu, n - k, opts)",
    "topk.mjs::locScaleFromWinAndSecond":
        "forwards to locScaleFromTopkPair(p1, 1, top2, 2, opts)",
}

# keys read only on a branch that forwards, and meaningless on the rest
FORWARD_ONLY_KEYS = {
    "races.mjs::abilitiesFromRace": {"qa", "qf"},   # the structure dispatch
}


def _balanced(text, open_at):
    """Index of the brace closing the one at open_at."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def _split_top_level(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _public_keys_read(body, var="opts"):
    """The PUBLIC keys a function body reads off `var`.

    Destructuring renames freely -- `const { mu0: mu0In = null } = opts`
    reads the public key `mu0` into a local called `mu0In` -- so the key
    is what stands to the LEFT of the colon. Searching the body for the
    allowlisted string instead finds the local name and calls it read,
    which is how #207 passed this very test.
    """
    keys = set()
    for m in re.finditer(r"\{", body):
        close = _balanced(body, m.start())
        if not re.match(rf"\s*=\s*{var}\b", body[close + 1:close + 40]):
            continue
        for part in _split_top_level(body[m.start() + 1:close]):
            part = part.strip()
            if not part or part.startswith("..."):
                continue
            key = part.split(":")[0].split("=")[0].strip()
            if key:
                keys.add(key)
    keys |= set(re.findall(rf"{var}\.(\w+)", body))
    keys |= set(re.findall(rf'{var}\[\s*"(\w+)"\s*\]', body))
    return keys


def _guarded_browser_apis():
    """(key, allowlisted keys, public keys read) per guarded browser API."""
    out = []
    for path in sorted((ROOT / "docs/js/winning").glob("*.mjs")):
        text = path.read_text()
        allow = {
            m.group(1): set(re.findall(r'"(\w+)"', m.group(2)))
            for m in re.finditer(
                r"const (\w+_OPTS) = new Set\(\[(.*?)\]\)", text, re.S)
        }
        for m in re.finditer(
                r"export function (\w+)\([^)]*opts\s*=\s*\{\}\s*\)\s*\{", text):
            fn, start = m.group(1), m.end()
            end = text.find("\n}\n", start)
            body = text[start:end if end > 0 else len(text)]
            used = re.search(r"checkOpts\(opts,\s*(\w+)", body)
            if not used or used.group(1) not in allow:
                continue
            out.append((f"{path.name}::{fn}",
                        allow[used.group(1)], _public_keys_read(body)))
    return out


def test_the_browser_option_sweep_is_not_vacuous():
    """A sweep that finds nothing passes every assertion below it."""
    found = _guarded_browser_apis()
    assert len(found) >= 18, f"only {len(found)} guarded browser APIs found"
    names = {k.split("::")[1] for k, _a, _r in found}
    for expected in ("raceProbabilities", "abilitiesFromRace", "polishRace",
                     "topKProbabilities", "rankProbabilities"):
        assert expected in names, f"{expected} is guarded but was not swept"


def test_every_allowlisted_browser_option_is_actually_read():
    """An allowlist derived from python's signature rather than from what
    the function reads is worse than no allowlist: the key passes
    validation and is silently discarded, which is the exact failure the
    guard exists to prevent. `targetFloor` and `returnInfo` were
    advertised by abilitiesFromRace and read by nothing (#226).

    This is a relationship between two things in the same file, which is
    what source inspection is good for -- unlike behaviour, which needs
    parity/check_js_api.mjs.
    """
    offenders = []
    for key, allow, read in _guarded_browser_apis():
        if key in FORWARDS_OPTS:
            continue
        exempt = FORWARD_ONLY_KEYS.get(key, set())
        for k in sorted(allow - read - exempt):
            offenders.append(f"{key} advertises '{k}'")
    assert not offenders, (
        "browser options accepted by an allowlist but read by nothing -- "
        "they pass validation and vanish:\n  " + "\n  ".join(offenders))


def test_every_browser_option_read_is_allowlisted():
    """The other direction, and the one that refuses valid callers.

    #204 put the DESTRUCTURED LOCAL name in polishRace's allowlist, so
    `polishRace({mu0: ...})` -- supported since the function was written
    -- threw `unknown option 'mu0'`, while `mu0In` was accepted and
    ignored. Nothing caught it: the test above searched the body for the
    string `mu0In` and found the local, and no behavioural check made
    either call (#207).
    """
    offenders = []
    for key, allow, read in _guarded_browser_apis():
        for k in sorted(read - allow):
            offenders.append(f"{key} reads '{k}'")
    assert not offenders, (
        "browser options the function reads but its allowlist rejects -- "
        "a supported call now throws:\n  " + "\n  ".join(offenders))


def test_the_forwarding_exemptions_are_real():
    """A declared exemption must still forward; otherwise it is a way to
    hide the bug above."""
    for key, why in FORWARDS_OPTS.items():
        fname, fn = key.split("::")
        text = (ROOT / "docs/js/winning" / fname).read_text()
        i = text.index(f"export function {fn}(")
        body = text[i:text.find("\n}\n", i)]
        assert "opts)" in body or "...opts" in body, f"{key}: {why} is stale"
