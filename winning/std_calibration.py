"""Deprecation shim: the classic lattice API lives in winning.classic.

This module IS winning.classic.std_calibration (module aliasing, so private
attributes and module state work unchanged); new imports should say so.
"""
import sys as _sys
import warnings as _warnings

import winning.classic.std_calibration as _real

_warnings.warn(
    "winning.std_calibration moved to winning.classic.std_calibration; this top-level alias "
    "remains for compatibility",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _real
