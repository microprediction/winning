"""Deprecation shim: the classic lattice API lives in winning.classic.

This module IS winning.classic.scipyinclusion (module aliasing, so private
attributes and module state work unchanged); new imports should say so.
"""
import sys as _sys
import warnings as _warnings

import winning.classic.scipyinclusion as _real

_warnings.warn(
    "winning.scipyinclusion moved to winning.classic.scipyinclusion; this top-level alias "
    "remains for compatibility",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _real
