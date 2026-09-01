"""Deprecation shim: the classic lattice API lives in winning.classic.

This module IS winning.classic.lattice_copula (module aliasing, so private
attributes and module state work unchanged); new imports should say so.
"""
import sys as _sys
import warnings as _warnings

import winning.classic.lattice_copula as _real

_warnings.warn(
    "winning.lattice_copula moved to winning.classic.lattice_copula; this top-level alias "
    "remains for compatibility",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _real
