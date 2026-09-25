"""Every browser allowlist must name the PUBLIC option keys.

`checkOpts(opts, SOME_OPTS, "api")` rejects any key not in `SOME_OPTS`,
so that list IS the public signature of a browser API. Nothing keeps it
honest: javascript destructuring renames freely, and

    const { mu0: mu0In = null } = opts;

reads the public key `mu0` into a local called `mu0In`. #204 put the
LOCAL name in the allowlist, so `polishRace({mu0: ...})` -- a supported
call -- was rejected as an unknown option, while `mu0In` was accepted and
silently ignored. That is #207. Twenty-one behavioural guard checks and
eighty-seven surface tests all passed, because none of them made either
call.

Two asymmetric failures, both silent in their own way:

* a key in the list that nothing reads   -> accepted and ignored;
* a key read that is not in the list     -> a valid call is refused.

The check is static because "accepted and ignored" cannot be told from
"accepted and used" by calling alone -- an option may legitimately not
move a particular fixture. `parity/check_js_api.mjs` covers the
behaviour; this covers the whole surface at once, which is how #207
escaped. Same template as tests/test_shape_contract.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "docs" / "js" / "winning"

# Options an API accepts on behalf of a path it delegates to, where the
# delegate is named explicitly rather than handed `opts`.
DELEGATED = {
    # qa/qf are structure-only quadrature settings. Both race verbs take
    # them and use them only inside `if (structure)`, which forwards to
    # structures.mjs; the plain factor path has no quadrature to set.
    ("raceProbabilities", "qa"),
    ("raceProbabilities", "qf"),
    ("abilitiesFromRace", "qa"),
    ("abilitiesFromRace", "qf"),
}


def _balanced(src: str, open_at: int) -> int:
    """Index of the brace closing the one at open_at."""
    depth = 0
    for i in range(open_at, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def _top_level_split(text: str) -> list[str]:
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


def _keys_read(body: str, var: str) -> set[str]:
    """Public keys the function reads off `var`, however it renames them."""
    keys: set[str] = set()
    for m in re.finditer(r"\{", body):
        end = _balanced(body, m.start())
        if not re.match(r"\s*=\s*" + var + r"\b", body[end + 1:end + 40]):
            continue
        for part in _top_level_split(body[m.start() + 1:end]):
            part = part.strip()
            if not part or part.startswith("..."):
                continue
            key = part.split(":")[0].split("=")[0].strip()
            if key:
                keys.add(key)
    keys |= set(re.findall(var + r"\.(\w+)", body))
    keys |= set(re.findall(var + r'\[\s*"(\w+)"\s*\]', body))
    return keys


def _guarded_apis():
    """(module, api, allowlisted keys, keys read, forwards wholesale)."""
    out = []
    for path in sorted(JS.glob("*.mjs")):
        src = path.read_text()
        lists = {
            m.group(1): set(re.findall(r'"([^"]+)"', m.group(2)))
            for m in re.finditer(
                r"const\s+(\w*OPTS)\s*=\s*(?:new Set\(\s*)?\[(.*?)\]", src, re.S
            )
        }
        heads = [
            (m.group(1), m.start())
            for m in re.finditer(r"^(?:export\s+)?function\s+(\w+)", src, re.M)
        ]
        for i, (_name, start) in enumerate(heads):
            end = heads[i + 1][1] if i + 1 < len(heads) else len(src)
            body = src[start:end]
            call = re.search(
                r'checkOpts\(\s*(\w+)\s*,\s*(\w+)\s*,\s*"([^"]+)"', body)
            if not call:
                continue
            var, listname, api = call.groups()
            if listname not in lists:
                continue
            # `f(..., opts)` or `{...opts}`: the callee validates the rest
            forwards = bool(
                re.search(r"\.\.\.\s*" + var + r"\b", body)
                or re.search(r"\w+\([^()]*\b" + var + r"\s*\)", body)
            )
            out.append((path.name, api, lists[listname],
                        _keys_read(body, var), forwards))
    return out


GUARDED = _guarded_apis()


def test_the_sweep_actually_found_the_browser_apis():
    """A discovery test: an empty sweep must fail, not pass vacuously."""
    assert len(GUARDED) >= 18, f"only found {len(GUARDED)} guarded APIs"
    names = {api for _m, api, _a, _r, _f in GUARDED}
    for expected in ("raceProbabilities", "abilitiesFromRace", "polishRace",
                     "topKProbabilities", "rankProbabilities"):
        assert expected in names, f"{expected} is guarded but was not swept"


@pytest.mark.parametrize(
    "module,api,allow,read,forwards", GUARDED,
    ids=[f"{a}" for _m, a, _al, _r, _f in GUARDED])
def test_allowlist_names_the_public_keys(module, api, allow, read, forwards):
    refused = sorted(k for k in read - allow)
    assert not refused, (
        f"{module}::{api} READS {', '.join(refused)} but the allowlist "
        f"rejects them, so a supported call now throws")
    if forwards:
        return          # the delegate validates whatever this one passes on
    ignored = sorted(k for k in allow - read
                     if (api, k) not in DELEGATED)
    assert not ignored, (
        f"{module}::{api} ALLOWS {', '.join(ignored)} but never reads them, "
        f"so those calls are accepted and silently ignored. If the key is "
        f"handled by a delegate named explicitly, add it to DELEGATED with "
        f"the reason.")
