"""One switch for the compiled kernels.

The numpy implementations are the spec; the rust kernels (fastrace) are a
drop-in acceleration that every module guards with a `_HAVE_RUST` flag.
This module centralizes the toggle:

    pip install winning            pure python (numpy/scipy only)
    pip install winning[fast]      adds the compiled fastrace wheel

    WINNING_PURE=1                 env var: ignore fastrace even if present
    winning.use_rust(False)        runtime: same, reversible
    winning.rust_active()          which path will run

Parity between the two paths is pinned by tests/test_rust_parity.py.
"""
import os


def _rust_modules():
    from winning.classic import lattice, lattice_calibration
    from winning.factor import races, blocks, topk, permutations, core
    from winning.methods import native
    from winning.alternatives import reprs
    return [lattice, lattice_calibration, races, blocks, topk, permutations,
            core, native, reprs]


def use_rust(enabled=True):
    """Turn the compiled kernels on or off for every module at once.

    Turning them on is a no-op when fastrace is not installed."""
    for mod in _rust_modules():
        mod._HAVE_RUST = bool(enabled) and mod._fastrace is not None and \
            getattr(mod, "_RUST_OK", True)


def rust_active():
    """True if any module will dispatch to fastrace."""
    return any(mod._HAVE_RUST for mod in _rust_modules())


def pure_requested_by_env():
    return os.environ.get("WINNING_PURE", "").strip() not in ("", "0")


def load_fastrace(*needs):
    """Import the compiled kernels for one module: (module, ok, active).

    The environment is consulted FIRST (#113): under WINNING_PURE the
    extension is never imported, so a broken wheel (ABI mismatch, missing
    dependent library) cannot take the package down when the user has
    asked for pure python. `ok` says the extension has every kernel the
    caller names; `active` is `ok` and not pure. use_rust() flips
    `active` at runtime and keeps `ok` as its ceiling."""
    if pure_requested_by_env():
        return None, False, False
    try:
        import fastrace as mod
    except ImportError:
        return None, False, False
    ok = all(hasattr(mod, name) for name in needs)
    return mod, ok, ok
