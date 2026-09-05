"""Deprecation shim: the density-agnostic research engine lives in
winning.research.

This module IS winning.research (module aliasing, so submodule imports
and module state work unchanged); new imports should say so. The name
change is honest labeling: this is research-grade machinery, distinct
from the separate thurstone package it was once vendored from.
"""
import sys as _sys
import warnings as _warnings

import winning.research as _real

_warnings.warn(
    "winning.thurstone moved to winning.research; this alias remains "
    "for compatibility",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _real
