"""The package-wide loading-shape contract, swept over every public verb.

WHY THIS FILE EXISTS. Issue #66 was a 1-D `V` that `np.atleast_2d` read
as `(1, n)` -- one contestant with n factors. It survived 462 tests and
775 verifier checks, and the post-mortem found three harness gaps:

  1. NO REPRESENTATION INVARIANCE. Every test spelled loadings `(n, 1)`
     or `(n, r)`, so no two spellings of the same loadings were ever
     compared. The verifier's invariances are all over VALUES (gauge,
     permutation, scale, team) -- none over equivalent spellings of one
     input, and no test anywhere asserted a shape contract.
  2. CONSTANT LOADINGS ARE A BLIND ORACLE. A constant column is
     gauge-fixed to zero, so a factor race that has silently degenerated
     to the independent race still looks right. The issue's own repro
     used `np.full(n, 0.2)`, which is why it produced a panic instead of
     a wrong number.
  3. THE COMPILED PATH WAS THE LUCKY ONE. `fastrace` panicked;
     pure-numpy agreed with `V=None` to 4e-15 and said nothing.

So this file does four things, and the fourth is the one that matters
most: `test_every_public_V_is_covered` DISCOVERS every public function
taking a `V` and fails if it is neither driven below nor exempted with a
reason. A new entry point cannot join the package without joining the
sweep.
"""
import importlib
import inspect
import pkgutil
import re
import warnings

import numpy as np
import pytest

import winning
from winning.shapes import as_idio, as_loadings, as_variance

# ---------------------------------------------------------------- fixtures
N = 5
MU = np.linspace(-0.45, 0.45, N)
# NON-CONSTANT on purpose: a constant column is gauge-fixed to zero and
# cannot tell the factor race from the independent one (gap 2 above).
V1 = np.array([0.80, -0.40, 0.20, 0.60, -0.50])
V2 = np.array([[0.9, -0.2], [0.1, 0.7], [-0.5, 0.3], [0.4, -0.6], [0.0, 0.2]])


def _rank1(n):
    return V1[:n] if n <= N else np.linspace(0.8, -0.5, n)


def _rank2(n):
    return np.column_stack([_rank1(n), np.roll(_rank1(n), 2) * 0.7])
D = np.linspace(0.80, 1.05, N)
P = np.array([0.34, 0.26, 0.18, 0.13, 0.09])
ORDER = np.array([2, 0, 4, 1, 3])
RNG = lambda: np.random.default_rng(0)                            # noqa: E731


def _spellings(V):
    """Every accepted way to write the SAME loadings."""
    A = np.asarray(V, float)
    A = A.reshape(-1, 1) if A.ndim == 1 else A
    out = {"contract (n, rank)": A, "transposed (rank, n)": A.T}
    if A.shape[1] == 1:
        out["bare vector (n,)"] = A.ravel()
    return out


def _flat(x):
    """EVERY float in whatever a verb returns, concatenated.

    Not element 0 of a tuple: `correlated_draws` returns (abilities,
    performances) and only the second moves with V, so unpacking the
    first would have made this sweep vacuous for that verb -- the same
    blind-oracle mistake the sweep exists to catch.
    """
    out = []

    def walk(v):
        if v is None or isinstance(v, (str, bool, type)):
            return
        if isinstance(v, dict):
            for key in sorted(v, key=str):
                walk(v[key])
            return
        if isinstance(v, (tuple, list)):
            for e in v:
                walk(e)
            return
        try:
            a = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            return
        if a.size and np.isfinite(a).all():
            out.append(a)

    walk(x)
    return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------- drivers
# name -> (callable(V) -> result, needs_a_bite_check)
DRIVERS = {}


def driver(*names, bites=True, n=N, requires=None):
    """Register a driver.

    `n` is the field size the verb's V is indexed by -- the team verbs
    take one loading row per TEAM, not per entity. `requires` names an
    optional third-party backend the verb needs; CI installs only
    `.[test]` (pytest, pandas, matplotlib), so a verb behind jax must
    SKIP there rather than fail -- and skip visibly, by name, so the
    hole is in the report rather than silently absent.
    """
    def deco(fn):
        for nm in names:
            DRIVERS[nm] = (fn, bites, n, requires)
        return fn
    return deco


def _driver(name):
    """The driver, with its optional backend enforced as a skip."""
    fn, bites, n, requires = DRIVERS[name]
    if requires is not None:
        pytest.importorskip(
            requires,
            reason=f"{name} needs {requires!r}, which is not installed "
                   f"(CI installs only .[test]); the shape contract for "
                   f"this verb is unverified here, not waived")
    return fn, bites, n


def _mod(path):
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


# --- winning.factor.races / permutations / polish / topk -----------------
@driver("winning.factor.races.race_probabilities")
def _d_race(V):
    return _mod("winning.factor.races.race_probabilities")(
        MU, V=V, D=D, points=257)


@driver("winning.factor.races.abilities_from_race",
        "winning.factor.races.calibrate_abilities")
def _d_abil(V):
    return _mod("winning.factor.races.abilities_from_race")(
        P, V=V, D=D, points=257)


@driver("winning.factor.races.removal_shares")
def _d_removal(V):
    return _mod("winning.factor.races.removal_shares")(MU, V=V, D=D, points=501)


@driver("winning.factor.races.tie_densities")
def _d_ties(V):
    return _mod("winning.factor.races.tie_densities")(MU, V=V, D=D, points=501)


@driver("winning.factor.races.softmax_probabilities")
def _d_softmax(V):
    return _mod("winning.factor.races.softmax_probabilities")(
        MU, temperature=0.7, V=V)


@driver("winning.factor.races.harville_order_logprob")
def _d_harville(V):
    return _mod("winning.factor.races.harville_order_logprob")(
        MU, ORDER, temperature=0.7, V=V)


@driver("winning.factor.races.plackett_luce_order_logprob")
def _d_pl(V):
    return _mod("winning.factor.races.plackett_luce_order_logprob")(
        MU, ORDER, temperature=0.7, V=V)


@driver("winning.factor.permutations.ordered_probabilities")
def _d_ordered(V):
    return _mod("winning.factor.permutations.ordered_probabilities")(
        MU, k=2, V=V, D=D, points=301)


@driver("winning.factor.permutations.harville_prefix_logprob")
def _d_hprefix(V):
    return _mod("winning.factor.permutations.harville_prefix_logprob")(
        MU, [2, 0, 4], temperature=0.7, V=V)


@driver("winning.factor.permutations.plackett_luce_prefix_logprob")
def _d_plprefix(V):
    return _mod("winning.factor.permutations.plackett_luce_prefix_logprob")(
        MU, [2, 0, 4], temperature=0.7, V=V)


@driver("winning.factor.polish.race_jacobian")
def _d_jac(V):
    return _mod("winning.factor.polish.race_jacobian")(
        MU, V=V, D=D, points=257)


@driver("winning.factor.polish.race_jacobian_row")
def _d_jacrow(V):
    return _mod("winning.factor.polish.race_jacobian_row")(
        MU, 1, V=V, D=D, points=257)


@driver("winning.factor.polish.polish_race")
def _d_polish(V):
    return _mod("winning.factor.polish.polish_race")(
        p0=P, mu0=MU, V=V, D=D, points=257)


@driver("winning.factor.topk.top_k_probabilities")
def _d_topk(V):
    return _mod("winning.factor.topk.top_k_probabilities")(
        MU, 2, V=V, D=D, points=257)


@driver("winning.factor.topk.bottom_k_probabilities")
def _d_botk(V):
    return _mod("winning.factor.topk.bottom_k_probabilities")(
        MU, 2, V=V, D=D, points=257)


@driver("winning.factor.topk.rank_probabilities")
def _d_rank(V):
    return _mod("winning.factor.topk.rank_probabilities")(
        MU, D=D, V=V, points=257)


@driver("winning.factor.topk.top_k_jacobians")
def _d_topkjac(V):
    return _mod("winning.factor.topk.top_k_jacobians")(
        MU, 2, D=D, V=V, points=257)


@driver("winning.factor.topk.abilities_from_topk")
def _d_abtopk(V):
    q = _mod("winning.factor.topk.top_k_probabilities")(
        MU, 2, V=V2, D=D, points=257)
    return _mod("winning.factor.topk.abilities_from_topk")(
        q, 2, V=V, D=D, points=257)


# --- winning.factor.core -------------------------------------------------
def _nodes(r):
    return _mod("winning.factor.core.hermite_nodes")(r, Q=11)


@driver("winning.factor.core.win_probabilities_factor")
def _d_wpf(V):
    Vc = as_loadings(V, N)
    F, W = _nodes(Vc.shape[1])
    return _mod("winning.factor.core.win_probabilities_factor")(MU, V, D, F, W)


@driver("winning.factor.core.abilities_from_probabilities_factor",
        "winning.factor.core.abilities_from_win_probabilities")
def _d_apf(V):
    Vc = as_loadings(V, N)
    F, W = _nodes(Vc.shape[1])
    return _mod("winning.factor.core.abilities_from_probabilities_factor")(
        P, V, D, F, W)


@driver("winning.factor.core.jacobian_vector_product")
def _d_jvp(V):
    Vc = as_loadings(V, N)
    F, W = _nodes(Vc.shape[1])
    # NOT h = ones: the Jacobian's rows sum to zero, so J @ 1 is
    # identically zero whatever V is, and the sweep would prove nothing
    return _mod("winning.factor.core.jacobian_vector_product")(
        MU, V, D, F, W, np.array([1.0, -0.5, 0.3, 0.0, -0.8]), points=501)


# --- winning.probit ------------------------------------------------------
@driver("winning.probit.shares")
def _d_pshares(V):
    return _mod("winning.probit.shares")(MU, V=V, D=D)


@driver("winning.probit.removal_shares")
def _d_premoval(V):
    return _mod("winning.probit.removal_shares")(MU, V=V, D=D)


@driver("winning.probit.utilities_from_shares",
        "winning.probit.calibrate_utilities")
def _d_putil(V):
    return _mod("winning.probit.utilities_from_shares")(P, V=V, D=D)


# --- winning.likelihood --------------------------------------------------
@driver("winning.likelihood.choice_loglik_and_score")
def _d_choice(V):
    return _mod("winning.likelihood.choice_loglik_and_score")(
        np.tile(MU, (3, 1)), V, np.array([0, 1, 2]), D=D)


@driver("winning.likelihood.sharpness_bound", bites=False)
def _d_sharpness(V):
    """Returns a scalar, so it cannot BITE the way a probability vector
    does; what matters is that it reads every spelling of V alike, which
    the sweep checks by comparing across them."""
    return np.array([_mod("winning.likelihood.sharpness_bound")(V, D)])


@driver("winning.likelihood.ranking_loglik_and_score")
def _d_ranking(V):
    return _mod("winning.likelihood.ranking_loglik_and_score")(
        np.tile(MU, (3, 1)), V, np.tile(ORDER, (3, 1)))


# --- winning.fastmvn -----------------------------------------------------
@driver("winning.fastmvn.mvn_cdf_fast", "winning.fastmvn.mvn_cdf_fast_info")
def _d_mvn(V):
    return _mod("winning.fastmvn.mvn_cdf_fast")(
        upper=np.full(N, 0.8), V=V, D=D)


# --- winning.methods (all twelve share one signature) --------------------
_METHODS = ["winning.methods.native." + m for m in
            ("lattice", "direct_mc", "sobol_direct", "factor_rqmc", "ghk",
             "qmc_ghk", "tilting", "stern")] + \
           ["winning.methods.orthant_extra." + m for m in
            ("genz_bretz", "mendell_elston", "ep_orthant", "smc_orthant")]

for _name in _METHODS:
    driver(_name)(lambda V, _n=_name: _mod(_n)(MU, V, D)[0])


# --- winning.alternatives ------------------------------------------------
@driver("winning.alternatives.cdf_grad.cdf_gradient_shares", requires="jax")
def _d_cdfgrad(V):
    return _mod("winning.alternatives.cdf_grad.cdf_gradient_shares")(MU, V, D)


@driver("winning.alternatives.reprs.per_winner_reduced_rank_shares")
def _d_perwinner(V):
    return _mod("winning.alternatives.reprs.per_winner_reduced_rank_shares")(
        MU, V, D)


@driver("winning.alternatives.reprs.reduced_rank_representation", bites=False)
def _d_reduced(V):
    return _mod("winning.alternatives.reprs.reduced_rank_representation")(
        MU, V, D, 0)


# --- winning.ratings -----------------------------------------------------
@driver("winning.ratings.nway.update_winner_correlated")
def _d_nwinner(V):
    return _mod("winning.ratings.nway.update_winner_correlated")(
        MU, D, 1, V, beta2=1.0)


@driver("winning.ratings.nway.update_order_correlated")
def _d_norder(V):
    return _mod("winning.ratings.nway.update_order_correlated")(
        MU, D, ORDER, V, beta2=1.0)


@driver("winning.ratings.nway.predictive_win_probabilities")
def _d_npred(V):
    return _mod("winning.ratings.nway.predictive_win_probabilities")(
        MU, D, V=V, beta2=1.0)


@driver("winning.ratings.market.update_race")
def _d_market(V):
    return _mod("winning.ratings.market.update_race")(
        MU, D, p_market=P, tau2=0.25, V=V, beta2=1.0)


@driver("winning.ratings.full.update_winner_full")
def _d_fwinner(V):
    return _mod("winning.ratings.full.update_winner_full")(
        MU, np.diag(D), 1, V=V, beta2=1.0)


@driver("winning.ratings.full.update_order_full")
def _d_forder(V):
    return _mod("winning.ratings.full.update_order_full")(
        MU, np.diag(D), ORDER, V=V, beta2=1.0)


@driver("winning.ratings.history.update_margins_full")
def _d_margins(V):
    return _mod("winning.ratings.history.update_margins_full")(
        MU, np.diag(D), margins=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        V=V, beta2=1.0)


@driver("winning.ratings.simulate.correlated_draws")
def _d_draws(V):
    return _mod("winning.ratings.simulate.correlated_draws")(
        RNG(), 64, MU, D, V=V)


@driver("winning.ratings.simulate.independent_world")
def _d_world(V):
    return _mod("winning.ratings.simulate.independent_world")(
        RNG(), M=N, K=3, n=6, V=V)


@driver("winning.ratings.history.predict_race")
def _d_predict(V):
    state = {"m": MU, "S": np.diag(D), "t": None,
             "index": {f"e{i}": i for i in range(N)},
             "prior_mean": 0.0, "prior_var": 1.0}
    return _mod("winning.ratings.history.predict_race")(
        state, [f"e{i}" for i in range(N)], V=V, beta2=1.0)


# team verbs: A is the line-up incidence, V is loadings per TEAM row
_A = np.array([[1.0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])
_VT = np.array([0.7, -0.3, 0.4])
_MT, _ST = np.array([0.2, -0.1, 0.05]), np.diag([0.9, 1.0, 1.1])


@driver("winning.ratings.teams.update_team_winner_full", n=3)
def _d_twinner(V):
    return _mod("winning.ratings.teams.update_team_winner_full")(
        MU, np.diag(D), _A, 1, V=V, beta2=1.0)


@driver("winning.ratings.teams.update_team_order_full", n=3)
def _d_torder(V):
    return _mod("winning.ratings.teams.update_team_order_full")(
        MU, np.diag(D), _A, np.array([2, 0, 1]), V=V, beta2=1.0)


@driver("winning.ratings.teams.update_team_margins_full", n=3)
def _d_tmargins(V):
    return _mod("winning.ratings.teams.update_team_margins_full")(
        MU, np.diag(D), _A, margins=np.array([0.0, 1.0, 2.0]), V=V, beta2=1.0)


# --------------------------------------------------------------- exemptions
# Not driven, each for a stated reason. Keep this list SHORT and specific:
# an entry here is a hole in the sweep, so it must say why.
EXEMPT = {
    "winning.shapes.as_loadings":
        "is the normaliser; its own contract is tested directly below",
    "winning.factor.races.forward_grid":
        "lattice helper: takes M_all (Q, n), not mu, so V is already "
        "gauge-fixed and shaped by _setup before it is reached",
    "winning.factor.topk.loc_scale_from_topk_pair":
        "REFUSES V with a documented NotImplementedError (fixing the "
        "loadings destroys the joint rescaling gauge and the two-curve "
        "fit is under-identified), so it has no spelling to be "
        "invariant over -- pinned by test_the_refusing_verb_still_refuses",
}


# -------------------------------------------------------------------- tests
def _discover_by_param(param):
    """Every public function in the package taking a parameter by name.

    Import failures are collected, not swallowed: a module that stops
    importing would otherwise drop out of discovery and shrink this
    sweep without anything failing -- a silent hole of exactly the kind
    that let #66 through.
    """
    found, unimportable = {}, {}
    for m in pkgutil.walk_packages(winning.__path__, "winning."):
        if ".research" in m.name or "verify" in m.name:
            continue
        try:
            mod = importlib.import_module(m.name)
        except Exception as exc:                       # noqa: BLE001
            unimportable[m.name] = f"{type(exc).__name__}: {exc}"
            continue
        for fn_name, fn in vars(mod).items():
            if fn_name.startswith("_") or not inspect.isfunction(fn):
                continue
            if getattr(fn, "__module__", None) != m.name:
                continue
            try:
                sig = inspect.signature(fn)
            except (ValueError, TypeError):
                continue
            if param in sig.parameters:
                found[f"{m.name}.{fn_name}"] = fn
    return found, unimportable


def _discover():
    return _discover_by_param("V")


# winning itself needs only numpy and scipy. Anything else a module
# imports is an optional backend -- jax, sklearn, trueskill, fastrace,
# pandas, matplotlib -- and a minimal install legitimately lacks it.
HARD_DEPS = ("numpy", "scipy")
# match the MESSAGE, not the class name: ModuleNotFoundError subclasses
# ImportError and either can carry a missing optional backend
_MISSING = re.compile(r"No module named '?([A-Za-z_][\w.]*)'?")


def _split_import_failures(unimportable):
    """Separate 'an optional backend is absent' from a REAL broken import.

    Keyed on the cause, not on a list of module names: the first version
    of this test allowlisted two paths I guessed at, neither of which
    existed, so `winning.bench.season_ranked` needing trueskill took CI
    down on five platforms while passing locally where trueskill happens
    to be installed. A name list cannot be kept true; the rule can.
    """
    # modules that refuse to import ON PURPOSE, with a message saying why
    TOMBSTONES = {"winning.thurstone": "removed; raises ImportError with "
                                       "migration guidance (see its source)"}
    optional, real = {}, {}
    for mod, err in unimportable.items():
        if mod in TOMBSTONES:
            continue
        m = _MISSING.search(err)
        missing = m.group(1).split(".")[0] if m else None
        if (missing and missing not in HARD_DEPS
                and not missing.startswith("winning")):
            optional[mod] = missing
        else:
            real[mod] = err
    return optional, real


def test_every_public_V_is_covered():
    """The gap that let #66 through: nothing forced a new entry point
    into the shape sweep. Now a `V` that is neither driven nor exempted
    fails here, at the moment it is added."""
    discovered, unimportable = _discover()
    optional, real = _split_import_failures(unimportable)
    assert not real, (
        "modules that no longer import, so their V-taking functions "
        "silently left this sweep:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(real.items())))
    if optional:
        # not a failure -- but say so, so the narrowed sweep is visible
        # in the run rather than inferred from a green tick
        warnings.warn(
            "shape sweep narrowed: these modules need an optional backend "
            "that is not installed, so their V-taking functions were not "
            "swept here: "
            + ", ".join(f"{k} (needs {v})" for k, v in sorted(optional.items())),
            UserWarning, stacklevel=2)
    found = set(discovered)
    covered = set(DRIVERS) | set(EXEMPT)
    missing = sorted(found - covered)
    assert not missing, (
        "public functions taking V that are neither driven nor exempted "
        "in tests/test_shape_contract.py:\n  " + "\n  ".join(missing))
    stale = sorted(covered - found)
    assert not stale, ("drivers/exemptions naming functions that no "
                       "longer exist:\n  " + "\n  ".join(stale))


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_every_spelling_of_the_same_loadings_agrees(name):
    """Representation invariance: the gap that had no test at all."""
    call, _, n = _driver(name)
    for V in (_rank1(n), _rank2(n)):
        spellings = _spellings(V)
        ref_key = "contract (n, rank)"
        ref = _flat(call(spellings[ref_key]))
        # A few ULP, not bitwise. as_loadings itself IS bitwise identical
        # across spellings (pinned separately below), but what a verb does
        # with the normalised matrix can reduce in a different order when
        # BLAS is in a different thread state, and that drifts by ~1 ULP
        # between processes regardless of how V was spelled. The bound is
        # still ~7e-15 against the 3.6e-2 error #66 produced.
        tol = 32 * np.finfo(float).eps * max(1.0, float(np.abs(ref).max()))
        for key, spelling in spellings.items():
            if key == ref_key:
                continue
            got = _flat(call(spelling))
            assert got.shape == ref.shape, f"{name}: {key} changed the shape"
            bad = np.abs(got - ref).max()
            assert bad <= tol, (
                f"{name}: {key} differs from {ref_key} by {bad:.3e} "
                f"(tolerance {tol:.1e})")


# the factor verbs are the ones with a compiled kernel behind them
_COMPILED = sorted(k for k in DRIVERS if k.startswith("winning.factor."))


@pytest.mark.parametrize("name", _COMPILED)
def test_every_spelling_agrees_on_the_pure_numpy_path_too(name, monkeypatch):
    """Gap 3: the compiled path was the LUCKY one. fastrace panicked on a
    1-D V while pure-numpy agreed with V=None to 4e-15 and said nothing,
    so a shape fix verified only against fastrace proves half of it.

    _HAVE_RUST is read per call, so setattr is enough; reloading the
    module to re-read WINNING_PURE rebinds every function in it and
    leaves other modules holding the old objects (that poisoned
    test_ratings_verify.py::test_seeds_are_order_and_worker_independent).
    Each of the four factor modules with a compiled path carries its own
    module-level flag (they import fastrace independently), so all four
    are patched.
    """
    for mod in ("winning.factor.races", "winning.factor.permutations",
                "winning.factor.topk", "winning.factor.blocks",
                "winning.factor.core"):
        # each of these four carries its OWN module-level _HAVE_RUST
        monkeypatch.setattr(importlib.import_module(mod), "_HAVE_RUST", False)
    test_every_spelling_of_the_same_loadings_agrees(name)


@pytest.mark.parametrize("name", sorted(k for k in DRIVERS if DRIVERS[k][1]))
def test_the_loadings_actually_move_the_answer(name):
    """The blind-oracle guard. Without this every equality above could
    hold because the verb ignores V -- which is exactly the state #66 put
    the factor race into."""
    call, _, n = _driver(name)
    a = _flat(call(_rank1(n).reshape(n, 1)))
    b = _flat(call(np.zeros((n, 1))))
    assert np.abs(a - b).max() > 1e-6, (
        f"{name}: rank-one loadings move the answer by less than "
        "1e-6 against zero loadings, so the invariance test above proves "
        "nothing about this verb")


@pytest.mark.parametrize("name", sorted(DRIVERS))
@pytest.mark.parametrize("bad", ["short", "cube", "unrelated"])
def test_a_mis_shaped_V_raises_and_names_the_contract(name, bad):
    """Every rejection is one ValueError with the same message -- not a
    panic, not `not enough values to unpack` three frames down."""
    call, _, n = _driver(name)
    V = {"short": np.full(n - 2, 0.3),
         "cube": np.full((2, n, 1), 0.3),
         "unrelated": np.full((n + 3, n + 2), 0.3)}[bad]
    with pytest.raises(ValueError, match=r"one row per contestant"):
        call(V)


def test_the_refusing_verb_still_refuses_every_spelling():
    """loc_scale_from_topk_pair is exempt from the sweep because it takes
    no correlation at all. Pin that, so the exemption cannot quietly
    become a hole if the refusal is ever lifted without a driver."""
    f = _mod("winning.factor.topk.loc_scale_from_topk_pair")
    q = np.array([0.5, 0.3, 0.12, 0.05, 0.03])
    for spelling in _spellings(V1).values():
        with pytest.raises(NotImplementedError, match=r"under-identified"):
            f(q, 1, q, 2, V=spelling)


def test_the_normaliser_is_bitwise_identical_across_spellings():
    """The exactness claim, at the one place it is ours to make: however
    the caller writes the loadings, the kernels receive the same bytes in
    the same C order. (Without ascontiguousarray the transposed spelling
    handed BLAS an F-ordered matrix and the answers parted at 1 ULP.)"""
    for V in (V1, V2):
        A = np.asarray(V, float)
        A = A.reshape(-1, 1) if A.ndim == 1 else A
        ref = as_loadings(A, N)
        assert ref.flags["C_CONTIGUOUS"]
        for spelling in _spellings(V).values():
            got = as_loadings(spelling, N)
            assert got.flags["C_CONTIGUOUS"]
            assert got.tobytes() == ref.tobytes()


def test_the_normaliser_accepts_exactly_the_documented_spellings():
    assert as_loadings(V1, N).shape == (N, 1)
    assert np.array_equal(as_loadings(V1, N).ravel(), V1)
    assert np.array_equal(as_loadings(V2, N), V2)
    assert np.array_equal(as_loadings(V2.T, N), V2)
    assert np.array_equal(as_loadings(0.4, 3), np.full((3, 1), 0.4))
    assert as_loadings(np.zeros((N, 0)), N).shape == (N, 0)   # empty = indep
    assert np.array_equal(as_idio(0.5, 3), np.full(3, 0.5))
    for bad in (np.full(N - 1, 0.3), np.full((2, N, 1), 0.3)):
        with pytest.raises(ValueError, match=r"one row per contestant"):
            as_loadings(bad, N)
    with pytest.raises(ValueError, match=r"one idiosyncratic variance"):
        as_idio(np.ones(N + 1), N)


# ---------------------------------------------------- silent-swap sweep
# (m, v, V) triples: mean, belief VARIANCE, loadings. At rank one all
# three are length-n float vectors, and `v` differs from `V` by case
# alone in five public signatures, so before as_variance existed any two
# of them could be exchanged without a complaint -- measured at 1.09 on
# the posterior mean, and a sqrt of a negative (NaN, warning only) in
# correlated_draws. as_variance makes the variance slot reject anything
# with a negative entry, and loadings are gauge-fixed to mean zero, so
# a non-trivial loading vector always has one.
_M = np.linspace(-0.45, 0.45, N)
_V_BELIEF = np.linspace(0.30, 0.55, N)
_LOAD = V1                                    # non-constant, has negatives

SWAPPABLE = {
    "nway.update_winner_correlated":
        lambda a, b: importlib.import_module(
            "winning.ratings.nway").update_winner_correlated(_M, a, 1, b),
    "nway.update_order_correlated":
        lambda a, b: importlib.import_module(
            "winning.ratings.nway").update_order_correlated(_M, a, ORDER, b),
    "nway.predictive_win_probabilities":
        lambda a, b: importlib.import_module(
            "winning.ratings.nway").predictive_win_probabilities(_M, a, V=b),
    "market.update_race":
        lambda a, b: importlib.import_module(
            "winning.ratings.market").update_race(_M, a, winner=1, V=b),
}


@pytest.mark.parametrize("name", sorted(SWAPPABLE))
def test_the_variance_and_the_loadings_cannot_be_exchanged(name):
    """The right way round works; the swap raises rather than answering
    a different question."""
    call = SWAPPABLE[name]
    call(_V_BELIEF, _LOAD)                       # correct order: fine
    with pytest.raises(ValueError, match=r"cannot be negative"):
        call(_LOAD, _V_BELIEF)                   # exchanged: refused


# every public verb taking a belief variance, and the value it must refuse
NEGATIVE_VARIANCE = {
    "winning.ratings.nway.update_winner":
        lambda v: importlib.import_module(
            "winning.ratings.nway").update_winner(_M, v, 1),
    "winning.ratings.nway.update_ranking":
        lambda v: importlib.import_module(
            "winning.ratings.nway").update_ranking(_M, v, ORDER),
    "winning.ratings.nway.update_ranking_exact":
        lambda v: importlib.import_module(
            "winning.ratings.nway").update_ranking_exact(_M, v, ORDER),
    "winning.ratings.nway.pairwise_update_winner":
        lambda v: importlib.import_module(
            "winning.ratings.nway").pairwise_update_winner(_M, v, 1),
    "winning.ratings.market.update_market":
        lambda v: importlib.import_module(
            "winning.ratings.market").update_market(
                _M, v, p_market=np.full(N, 1.0 / N)),
    "winning.ratings.simulate.correlated_draws":
        lambda v: importlib.import_module(
            "winning.ratings.simulate").correlated_draws(
                RNG(), 32, _M, v, V=_LOAD),
    "winning.ratings.tracker.order_augmented":
        lambda v: importlib.import_module(
            "winning.ratings.tracker").order_augmented(
                _M, v, ORDER, 1.0, RNG()),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE_VARIANCE))
def test_a_negative_belief_variance_is_refused(name):
    """A variance of zero is legal (perfectly known); negative never is,
    and accepting one is what let a loading vector through."""
    call = NEGATIVE_VARIANCE[name]
    call(_V_BELIEF)                              # a real variance: fine
    with pytest.raises(ValueError, match=r"cannot be negative"):
        call(-np.abs(_V_BELIEF))


def test_a_round_off_negative_variance_is_clipped_not_refused():
    """The variance check is a TOLERANCE, not `>= 0`.

    A symmetric eigendecomposition routinely leaves a -1e-18 where the
    true value is zero, and fit_covariance returns idiosyncratic
    variances down at 1e-6, so a strict test would raise on a caller
    doing nothing wrong. Loading entries are O(0.1-1) once gauge-fixed,
    so a relative 1e-12 still separates the two by twelve orders.
    """
    for ok in ([0.5, 0.3, -1e-18], [1e-6, 2e-6, -3e-19],
               [0.0, 0.0, 0.0], [1e6, 1e6, -1e-7]):
        got = as_variance(np.array(ok), 3)
        assert got.min() == 0.0 or got.min() > 0.0
        assert np.isfinite(got).all()
    for bad in ([0.8, -0.4, 0.2], [-1e-9, 0.5, 0.5], [1.0, 1.0, -1e-11]):
        with pytest.raises(ValueError, match=r"cannot be negative"):
            as_variance(np.array(bad), 3)
    # clipping must not write through to the caller's array
    src = np.array([0.5, 0.3, -1e-18])
    as_variance(src, 3)
    assert src[2] == -1e-18
    with pytest.raises(ValueError, match=r"must be finite"):
        as_variance(np.array([0.5, np.nan, 0.2]), 3)


def test_every_verb_taking_a_belief_variance_is_swept():
    """Same discipline as the V sweep: a new verb with a `v` parameter
    must be driven above, or this fails."""
    swept = set(NEGATIVE_VARIANCE) | {
        "winning.ratings.nway." + k.split(".")[-1] for k in SWAPPABLE
        if k.startswith("nway.")} | {
        "winning.ratings.market." + k.split(".")[-1] for k in SWAPPABLE
        if k.startswith("market.")}
    exempt = {
        # the normaliser itself; its contract is tested directly
        "winning.shapes.as_variance",
        # pure algebra on an already-validated state, no field to check
        "winning.ratings.history.diffuse",
        "winning.ratings.history.diffuse_full",
        # take prior_var (a scalar prior), not a per-entity variance
        "winning.ratings.history.rate_history",
        "winning.ratings.simulate.independent_world",
    }
    discovered, _ = _discover_by_param("v")
    missing = sorted(set(discovered) - swept - exempt)
    assert not missing, (
        "public verbs taking a belief variance that no swap/negativity "
        "test drives:\n  " + "\n  ".join(missing))


# ------------------------------------------------- the pair (n == 2) sweep
# The drivers above all use a 5-runner field, so the n == 2 path -- which
# `race_probabilities` now answers with a closed form rather than the
# lattice -- was covered only by checks run BY HAND when that landed.
# Hand-checking is what this file exists to replace, so the pair gets the
# same three contract checks as every other field size.
#
# A pair is the size at which the shapes are most confusable: at rank one
# V is (2, 1), D is (2,), mu is (2,) and the belief variance is (2,) --
# four arguments of identical shape.
PAIR_MU = np.array([0.15, -0.15])
PAIR_V = np.array([0.7, -0.3])          # non-constant, so it moves the race
PAIR_D = np.array([0.9, 1.1])
PAIR_P = np.array([0.6, 0.4])
PAIR_ORDER = np.array([1, 0])


def _pair(mod, attr):
    return getattr(importlib.import_module(mod), attr)


# Degenerate at a pair, so deliberately absent from PAIR_DRIVERS:
#   removal_shares -- removing one of two runners leaves one who wins with
#   probability 1, so the answer is the constant permutation matrix
#   [[0,1],[1,0]] and V cannot enter it. Verified, not assumed: the
#   loadings-move-the-answer check below failed on it, which is the guard
#   doing its job rather than a defect in the verb.
PAIR_DRIVERS = {
    "races.race_probabilities":
        lambda V: _pair("winning.factor.races", "race_probabilities")(
            PAIR_MU, V=V, D=PAIR_D, points=257),
    "races.abilities_from_race":
        lambda V: _pair("winning.factor.races", "abilities_from_race")(
            PAIR_P, V=V, D=PAIR_D, points=257),
    "races.tie_densities":
        lambda V: _pair("winning.factor.races", "tie_densities")(
            PAIR_MU, V=V, D=PAIR_D, points=501),
    "polish.race_jacobian":
        lambda V: _pair("winning.factor.polish", "race_jacobian")(
            PAIR_MU, V=V, D=PAIR_D, points=257),
    "polish.race_jacobian_row":
        lambda V: _pair("winning.factor.polish", "race_jacobian_row")(
            PAIR_MU, 1, V=V, D=PAIR_D, points=257),
    "permutations.ordered_probabilities":
        lambda V: _pair("winning.factor.permutations",
                        "ordered_probabilities")(
            PAIR_MU, k=2, V=V, D=PAIR_D, points=301),
    "topk.top_k_probabilities":
        lambda V: _pair("winning.factor.topk", "top_k_probabilities")(
            PAIR_MU, 1, V=V, D=PAIR_D, points=257),
    "probit.shares":
        lambda V: _pair("winning.probit", "shares")(
            PAIR_MU, V=V, D=PAIR_D),
    "nway.update_winner_correlated":
        lambda V: _pair("winning.ratings.nway", "update_winner_correlated")(
            PAIR_MU, PAIR_D, 1, V),
    "market.update_race":
        lambda V: _pair("winning.ratings.market", "update_race")(
            PAIR_MU, PAIR_D, p_market=PAIR_P, V=V),
}


@pytest.mark.parametrize("name", sorted(PAIR_DRIVERS))
def test_pair_every_spelling_of_the_same_loadings_agrees(name):
    call = PAIR_DRIVERS[name]
    ref = _flat(call(PAIR_V.reshape(2, 1)))
    tol = 32 * np.finfo(float).eps * max(1.0, float(np.abs(ref).max()))
    for key, spelling in (("bare vector (n,)", PAIR_V),
                          ("transposed (rank, n)", PAIR_V.reshape(1, 2))):
        got = _flat(call(spelling))
        assert got.shape == ref.shape, f"{name}: {key} changed the shape"
        bad = np.abs(got - ref).max()
        assert bad <= tol, f"{name}: {key} differs by {bad:.3e}"


@pytest.mark.parametrize("name", sorted(PAIR_DRIVERS))
def test_pair_the_loadings_actually_move_the_answer(name):
    """At n == 2 the blind-oracle risk is sharpest: a pair with zero
    loadings is still a perfectly well-formed race, so an equality that
    holds for the wrong reason looks entirely normal."""
    call = PAIR_DRIVERS[name]
    a = _flat(call(PAIR_V.reshape(2, 1)))
    b = _flat(call(np.zeros((2, 1))))
    assert np.abs(a - b).max() > 1e-6, (
        f"{name}: loadings do not move the pair, so the agreement above "
        "proves nothing about this verb")


@pytest.mark.parametrize("name", sorted(PAIR_DRIVERS))
@pytest.mark.parametrize("bad", ["short", "cube"])
def test_pair_a_mis_shaped_V_raises_and_names_the_contract(name, bad):
    call = PAIR_DRIVERS[name]
    V = {"short": np.full(4, 0.3), "cube": np.full((2, 2, 1), 0.3)}[bad]
    with pytest.raises(ValueError, match=r"one row per contestant"):
        call(V)


def test_pair_every_factor_verb_that_accepts_two_runners_is_swept():
    """Guard the guard: if a verb above stops accepting a pair, this says
    so rather than letting the entry quietly become dead weight."""
    for name, call in sorted(PAIR_DRIVERS.items()):
        out = _flat(call(PAIR_V.reshape(2, 1)))
        assert out.size, f"{name}: returned nothing usable at n == 2"


def test_no_module_reimplements_the_contract_with_atleast_2d():
    """The static half. `np.atleast_2d` on a loadings variable is the
    exact construct that caused #66: it turns (n,) into (1, n) silently.
    Loadings shapes go through winning.shapes, so this pattern must not
    come back anywhere in the package."""
    import pathlib
    root = pathlib.Path(winning.__file__).parent
    pat = re.compile(r"atleast_2d\(\s*np\.asarray\(\s*(V|Vm|Vv|loadings)\b")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "research" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{path.relative_to(root.parent)}:{i}: {line.strip()}")
    assert not offenders, (
        "loadings must be normalised by winning.shapes.as_loadings, not "
        "np.atleast_2d (which reads (n,) as (1, n)):\n  "
        + "\n  ".join(offenders))
