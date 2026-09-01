"""Deprecation shim: the classic lattice API lives in winning.classic.

This module IS winning.classic.pandas_util (module aliasing, so private
attributes and module state work unchanged); new imports should say so.
"""
import sys as _sys
import warnings as _warnings

import winning.classic.pandas_util as _real

_warnings.warn(
    "winning.pandas_util moved to winning.classic.pandas_util; this top-level alias "
    "remains for compatibility",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _real
