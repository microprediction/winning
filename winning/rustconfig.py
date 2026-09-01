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
    from winning.factor import races, blocks
    return [lattice, lattice_calibration, races, blocks]


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
