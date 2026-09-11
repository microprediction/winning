"""The verifier's machinery: a registry of checks, a Result per check
or per cell, deterministic seeding, a runner with an optional process
pool, and JSON / markdown reports with provenance.

A check is a function of a Context returning one Result or a list of
them. Every Result carries a verdict from VERDICTS:

  ok               the statistic met the mark in force
  FAIL             it did not (or the check raised; the type is in detail)
  UNDERPOWERED     the cell could not decide (too few seeds or draws);
                   distinct from ok on purpose: a thin cell reading as a
                   pass is how a load-dependent gate hid for a week
  EXPECTED_APPROX  outside the mark but inside a documented envelope
                   (marks.EXPECTED_APPROX), printed with its number
  SKIP             not applicable (base not supported, redundant cell),
                   with the reason -- the row stays visible
  MEASURED         a registered check with no mark yet: reported, never
                   gates (the "measure" half of measure -> adjudicate -> fix)

Marks live in marks.py and nowhere else; a check reads its tolerance
through ctx.mark() and cannot run with an ad-hoc number. Seeds derive
from ROOT_SEED and the check (and cell) name, so results are independent
of execution order, subset and worker count.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import marks

VERDICTS = ("ok", "FAIL", "UNDERPOWERED", "EXPECTED_APPROX", "SKIP", "MEASURED")
PROFILES = ("smoke", "fast", "full", "exhaustive")
_GATING = ("FAIL",)


@dataclass
class Result:
    name: str
    group: str
    verdict: str
    statistic: Optional[float] = None
    se: Optional[float] = None
    tolerance: Optional[float] = None
    detail: str = ""
    regime: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)
    n: Optional[int] = None
    runtime_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if self.verdict in ("EXPECTED_APPROX", "SKIP") and not self.detail:
            raise ValueError(f"{self.name}: {self.verdict} needs a reason in detail")

    def to_dict(self):
        d = asdict(self)
        for k in ("statistic", "se", "tolerance"):
            if d[k] is not None:
                d[k] = float(d[k])
        d["extras"] = _jsonable(d["extras"])
        d["regime"] = _jsonable(d["regime"])
        d["seeds"] = [int(s) for s in d["seeds"]]
        return d


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


@dataclass
class Check:
    name: str
    fn: Callable
    profiles: frozenset
    group: str
    cost_s: float


CHECKS: Dict[str, Check] = {}


def check(name, profiles=("full",), group=None, cost_s=1.0):
    """Register a check. profiles: the profiles it belongs to; a check in
    'smoke' is implicitly in every larger profile, 'fast' in full and
    exhaustive, 'full' in exhaustive."""
    prof = set(profiles)
    for p in prof:
        if p not in PROFILES:
            raise ValueError(f"{name}: unknown profile {p!r}")
    if "smoke" in prof:
        prof |= {"fast", "full", "exhaustive"}
    if "fast" in prof:
        prof |= {"full", "exhaustive"}
    if "full" in prof:
        prof |= {"exhaustive"}

    def deco(fn):
        if name in CHECKS:
            raise ValueError(f"duplicate check name {name!r}")
        CHECKS[name] = Check(name, fn, frozenset(prof),
                             group or name.split(".")[0], float(cost_s))
        return fn
    return deco


def seed_for(key, root=None):
    root = marks.ROOT_SEED if root is None else int(root)
    h = hashlib.sha256(f"{root}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little")


class Context:
    """What a check sees: its profile, its name, the marks module, the
    profile's parameters, and per-cell seeded generators."""

    def __init__(self, name, profile, seed_root=None):
        self.name = name
        self.profile = profile
        self.seed_root = marks.ROOT_SEED if seed_root is None else int(seed_root)
        self.marks = marks
        self.params = marks.PARAMS[profile]
        self.consumed: List[int] = []

    def seed(self, sub=""):
        s = seed_for(f"{self.name}:{sub}", self.seed_root)
        self.consumed.append(s)
        return s

    def rng(self, sub=""):
        return np.random.default_rng(self.seed(sub))

    def param(self, key, default=None):
        return self.params.get(key, default)

    def mark(self, key=None):
        """The mark entry for this check (or a sub-key), or None when the
        check is MEASURED."""
        k = self.name if key is None else key
        return marks.MARKS.get(k)

    def tolerance(self, key=None):
        m = self.mark(key)
        return None if m is None else m.get("tolerance")

    def verdict(self, statistic, key=None, se=None, k=None):
        """The standard decision: no mark -> MEASURED; |statistic| <= tol
        (+ k*se when given) -> ok; else EXPECTED_APPROX if inside the
        documented envelope; else FAIL. Returns (verdict, tolerance)."""
        m = self.mark(key)
        if m is None:
            return "MEASURED", None
        tol = float(m["tolerance"])
        if se is not None and k is not None:
            tol = tol + float(k) * float(se)
        if abs(float(statistic)) <= tol:
            return "ok", tol
        env = marks.EXPECTED_APPROX.get(self.name if key is None else key)
        if env is not None and abs(float(statistic)) <= float(env["envelope"]):
            return "EXPECTED_APPROX", tol
        return "FAIL", tol


def _run_one(name, profile, seed_root):
    chk = CHECKS[name]
    ctx = Context(name, profile, seed_root)
    t0 = time.time()
    try:
        out = chk.fn(ctx)
    except NotImplementedError as e:
        out = Result(name, chk.group, "SKIP", detail=f"not supported: {e}")
    except Exception as e:  # noqa: BLE001 -- a raising check is a failing check
        out = Result(name, chk.group, "FAIL",
                     detail=f"raised {type(e).__name__}: {str(e)[:200]}",
                     extras={"traceback": traceback.format_exc()[-2000:]})
    dt = time.time() - t0
    results = list(out) if isinstance(out, (list, tuple)) else [out]
    for r in results:
        if not r.seeds:
            r.seeds = list(ctx.consumed)
        if r.runtime_s == 0.0:
            r.runtime_s = dt / max(len(results), 1)
    return results


def _selected(profile, only=None):
    names = [n for n, c in CHECKS.items() if profile in c.profiles]
    if only:
        import fnmatch
        pats = [only] if isinstance(only, str) else list(only)
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    return sorted(names)


@dataclass
class Report:
    profile: str
    results: List[Result]
    provenance: dict
    wall_s: float

    def _with(self, verdict):
        return [r for r in self.results if r.verdict == verdict]

    @property
    def failed(self):
        return self._with("FAIL")

    @property
    def underpowered(self):
        return self._with("UNDERPOWERED")

    @property
    def expected_approx(self):
        return self._with("EXPECTED_APPROX")

    @property
    def measured(self):
        return self._with("MEASURED")

    @property
    def skipped(self):
        return self._with("SKIP")

    @property
    def ok(self):
        return not self.failed and not self.underpowered

    @property
    def exit_code(self):
        if self.failed:
            return 1
        if self.underpowered:
            return 2
        return 0

    def counts(self):
        return {v: sum(1 for r in self.results if r.verdict == v) for v in VERDICTS}

    def summary(self):
        c = self.counts()
        return (f"{self.profile}: {c['ok']} ok, {c['FAIL']} FAIL, "
                f"{c['UNDERPOWERED']} UNDERPOWERED, {c['EXPECTED_APPROX']} "
                f"EXPECTED_APPROX, {c['SKIP']} SKIP, {c['MEASURED']} MEASURED, "
                f"{self.wall_s / 60:.1f} min, {self.provenance.get('workers')} workers")

    def to_dict(self):
        return {"schema": 1, "profile": self.profile, "wall_s": self.wall_s,
                "provenance": _jsonable(self.provenance),
                "counts": self.counts(),
                "results": [r.to_dict() for r in self.results]}

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    def to_markdown(self):
        p = self.provenance
        lines = [f"# ratings verify: {self.profile}", "",
                 f"{self.summary()}", "",
                 f"winning {p.get('version')} at {p.get('git_sha') or 'wheel'}; "
                 f"{p.get('date')}; python {p.get('python')}, numpy {p.get('numpy')}, "
                 f"scipy {p.get('scipy')}, {p.get('platform')}; ROOT_SEED {p.get('root_seed')}, "
                 f"CONFIG_SEED {p.get('config_seed')}, marks {p.get('marks_sha256', '')[:12]}; "
                 f"threads {p.get('threads')}", ""]
        groups = sorted({r.group for r in self.results})
        for g in groups:
            lines += [f"## {g}", "",
                      "| check | regime | statistic | se | mark | verdict | n | s |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in self.results:
                if r.group != g:
                    continue
                reg = ",".join(f"{k}={v}" for k, v in r.regime.items()) if r.regime else ""
                st = "" if r.statistic is None else f"{r.statistic:.3g}"
                se = "" if r.se is None else f"{r.se:.2g}"
                tol = "" if r.tolerance is None else f"{r.tolerance:.3g}"
                nn = "" if r.n is None else str(r.n)
                lines.append(f"| {r.name} | {reg} | {st} | {se} | {tol} | {r.verdict} "
                             f"| {nn} | {r.runtime_s:.1f} |")
            lines.append("")
        notes = [r for r in self.results
                 if r.verdict in ("EXPECTED_APPROX", "SKIP", "FAIL", "UNDERPOWERED", "MEASURED")]
        if notes:
            lines += ["## Notes", ""]
            for r in notes:
                lines.append(f"- {r.name} [{r.verdict}]: {r.detail}")
            lines.append("")
        return "\n".join(lines)


def _git_sha():
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and len(sha) == 40 else None
    except Exception:  # noqa: BLE001
        return None


def _provenance(profile, workers, seed_root):
    import scipy
    import winning
    src = open(marks.__file__, "rb").read()
    return {"version": getattr(winning, "__version__", None),
            "git_sha": _git_sha(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "profile": profile, "workers": workers,
            "root_seed": marks.ROOT_SEED if seed_root is None else int(seed_root),
            "config_seed": marks.CONFIG_SEED,
            "marks_sha256": hashlib.sha256(src).hexdigest(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def verify(profile="fast", only=None, workers=1, verbose=True, out=None,
           seed_root=None):
    """Run every check registered for `profile` (optionally filtered by
    glob(s) in `only`), serially or on a spawn-context process pool, and
    return a Report. `out` is a directory: writes <date>_<sha7|wheel>_
    <profile>.json and .md there."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
    names = _selected(profile, only)
    t0 = time.time()
    results: List[Result] = []
    if workers and int(workers) > 1 and len(names) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=int(workers),
                                    mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_run_one, n, profile, seed_root): n for n in names}
            for fut in cf.as_completed(futs):
                rs = fut.result()
                results.extend(rs)
                if verbose:
                    for r in rs:
                        print(_line(r), flush=True)
    else:
        for n in names:
            rs = _run_one(n, profile, seed_root)
            results.extend(rs)
            if verbose:
                for r in rs:
                    print(_line(r), flush=True)
    results.sort(key=lambda r: r.name)
    rep = Report(profile, results, _provenance(profile, workers, seed_root),
                 time.time() - t0)
    if verbose:
        print(rep.summary(), flush=True)
    if out:
        os.makedirs(out, exist_ok=True)
        sha = (rep.provenance.get("git_sha") or "wheel")[:7]
        stem = f"{time.strftime('%Y-%m-%d', time.gmtime())}_{sha}_{profile}"
        rep.to_json(os.path.join(out, stem + ".json"))
        with open(os.path.join(out, stem + ".md"), "w") as f:
            f.write(rep.to_markdown())
    return rep


def _line(r):
    st = "" if r.statistic is None else f" {r.statistic:.3g}"
    tol = "" if r.tolerance is None else f" (mark {r.tolerance:.3g})"
    return f"  {r.verdict:15s} {r.name}{st}{tol}  {r.detail[:60]}"
