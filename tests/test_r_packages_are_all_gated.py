"""Every R package in r/ is checked by CI, and the parity harness runs.

The R workflow was path-triggered by changes under three packages and
then always built and checked exactly one of them, so a change confined
to either of the other two produced a GREEN check that never loaded the
changed package. It ran on pushes to main only, so no pull request was
gated. And `r/winning` -- the largest R package here, ten testthat files
-- was gated by nothing at all (#142).

A tabulated prime list survived in three of these packages at once
(#143), each with a silent cliff in a public rank argument, because
nothing ran them.

A workflow matrix cannot notice a package nobody added to it, so this
test is the thing that notices.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "r-check-fastpkgs.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _packages_on_disk():
    """Directories under r/ that are real R packages, not build output."""
    return {p.name for p in (ROOT / "r").iterdir()
            if p.is_dir() and (p / "DESCRIPTION").is_file()
            and not p.name.endswith(".Rcheck")}


def _packages_in_matrix():
    text = WORKFLOW.read_text()
    m = re.search(r"pkg:\s*\[([^\]]+)\]", text)
    assert m, "the workflow no longer declares a package matrix"
    return {p.strip() for p in m.group(1).split(",") if p.strip()}


def test_there_are_r_packages_to_gate():
    """Guards against every assertion below passing on an empty set."""
    found = _packages_on_disk()
    assert len(found) >= 4, f"only found {sorted(found)}"
    assert "winning" in found


def test_every_r_package_is_in_the_check_matrix():
    on_disk, in_matrix = _packages_on_disk(), _packages_in_matrix()
    missing = sorted(on_disk - in_matrix)
    assert not missing, (
        f"R packages that CI never builds or checks: {', '.join(missing)}. "
        f"Add them to the matrix in {WORKFLOW.name}.")
    stale = sorted(in_matrix - on_disk)
    assert not stale, (
        f"the matrix names packages that are not in r/: {', '.join(stale)}")


def test_the_r_workflow_runs_on_pull_requests():
    text = WORKFLOW.read_text()
    on = text.split("jobs:")[0]
    assert "pull_request:" in on, (
        "the R workflow runs on pushes only, so no pull request is gated")


def test_the_r_workflow_watches_all_of_r():
    """A per-package path list is how the old one skipped a package."""
    text = WORKFLOW.read_text()
    assert '"r/**"' in text or "'r/**'" in text, (
        "the workflow should watch all of r/, not a hand-listed subset")


def test_the_r_parity_harness_is_a_required_job():
    text = CI.read_text()
    assert "parity/check.R" in text, (
        "parity/check.R is the cross-language check for the R port and "
        "must run in CI")

def test_every_r_package_can_actually_run_its_tests():
    """Being in the matrix is not enough.

    `R CMD check` runs `tests/*.R`, and testthat needs a driver there --
    `library(testthat); library(pkg); test_check("pkg")` -- to reach
    `tests/testthat/`. Three of the four packages had testthat FILES and
    no driver, so their tests were dead weight: adding the packages to
    CI would have gated nothing but the build.
    """
    missing = []
    for pkg in sorted(_packages_on_disk()):
        d = ROOT / "r" / pkg / "tests"
        if not (d / "testthat").is_dir():
            continue                       # no testthat tests to drive
        driver = d / "testthat.R"
        if not driver.is_file():
            missing.append(f"{pkg}: tests/testthat/ exists, {driver.name} does not")
            continue
        text = driver.read_text()
        if f'test_check("{pkg}")' not in text:
            missing.append(f"{pkg}: {driver.name} does not test_check({pkg!r})")
    assert not missing, (
        "R packages whose testthat tests R CMD check will never run:\n  "
        + "\n  ".join(missing))


def test_no_r_test_sources_the_package_by_relative_path():
    """`source(file.path("..", "..", "R", ...))` works from the repo and
    fails inside a check, where the package is installed and there is no
    ../../R. It is also the wrong thing to test: under `test_check` the
    package NAMESPACE is already attached, internals included, so a test
    that sources the files checks a copy of the source rather than the
    thing that ships."""
    offenders = []
    for pkg in sorted(_packages_on_disk()):
        d = ROOT / "r" / pkg / "tests" / "testthat"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.R")):
            for num, line in enumerate(f.read_text().splitlines(), 1):
                if re.match(r"\s*source\(file\.path\(", line):
                    offenders.append(
                        f"{pkg}/{f.name}:{num}: {line.strip()[:56]}")
    assert not offenders, (
        "R tests that source the package instead of using the installed "
        "one:\n  " + "\n  ".join(offenders))

# Packages R ships with. Everything else must be installed by the
# workflow before `R CMD check --as-cran` runs, or the check ERRORS with
# "Packages suggested but not available".
_R_BASE = {
    "stats", "utils", "methods", "graphics", "grDevices", "tools",
    "parallel", "compiler", "datasets", "grid", "splines", "stats4",
    "tcltk", "translations", "base",
}


def _declared_deps(pkg):
    """Imports, Depends and Suggests for one package, as a set."""
    text = (ROOT / "r" / pkg / "DESCRIPTION").read_text(encoding="utf-8")
    fields, current = {}, None
    for line in text.splitlines():
        if re.match(r"^[A-Za-z/]+:", line):
            current, _, rest = line.partition(":")
            fields[current] = rest
        elif current and line.startswith((" ", "\t")):
            fields[current] += " " + line
    names = set()
    for key in ("Imports", "Depends", "Suggests"):
        for part in fields.get(key, "").split(","):
            name = re.sub(r"\(.*?\)", "", part).strip()
            if name and name != "R":
                names.add(name)
    return names - _R_BASE


def _workflow_installs():
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"install\.packages\(c\((.*?)\)\)", text, re.S)
    assert m, "the workflow no longer installs R packages in one call"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_ci_installs_every_declared_r_dependency():
    """`R CMD check --as-cran` ERRORS on a SUGGESTED package that is not
    installed, so a dependency the workflow does not install fails the
    job -- and it passes locally, where the developer has it.

    `mlogitfast` suggested `mlogit` and `dfidx` and used them nowhere:
    not in the code, the tests, the examples or a vignette, only named in
    prose on a manual page. The check passed here and failed on the
    runner with "Packages suggested but not available" (#142).
    """
    installs = _workflow_installs()
    missing = []
    for pkg in sorted(_packages_on_disk()):
        for dep in sorted(_declared_deps(pkg) - installs):
            missing.append(f"{pkg} declares {dep}")
    assert not missing, (
        "R dependencies the workflow never installs, so R CMD check "
        "errors on the runner while passing locally:\n  "
        + "\n  ".join(missing)
        + "\n\nEither add them to the install step, or drop them from "
          "DESCRIPTION if nothing actually uses them.")


def test_no_package_declares_a_dependency_nothing_uses():
    """The other half: a Suggests entry that no code, test or example
    references is dead weight that only CI pays for."""
    unused = []
    for pkg in sorted(_packages_on_disk()):
        root = ROOT / "r" / pkg
        body = ""
        for sub in ("R", "tests", "vignettes"):
            d = root / sub
            if d.is_dir():
                for f in d.rglob("*.R"):
                    body += f.read_text(encoding="utf-8", errors="ignore")
        for dep in sorted(_declared_deps(pkg)):
            if dep == "testthat":
                continue          # the driver names it
            if not re.search(rf"\b{re.escape(dep)}\b", body):
                unused.append(f"{pkg} declares {dep}, which no R code uses")
    assert not unused, (
        "declared R dependencies that nothing references:\n  "
        + "\n  ".join(unused))
