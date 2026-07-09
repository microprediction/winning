"""Deprecated winning 1.x module preserved for import compatibility.

The constants now live in thurstone.conventions.
"""

import warnings

from thurstone.conventions import (  # noqa: F401
    ALT_A,
    ALT_L,
    ALT_SCALE,
    ALT_UNIT,
    NAN_DIVIDEND,
    STD_A,
    STD_L,
    STD_SCALE,
    STD_UNIT,
)

warnings.warn(
    "winning.lattice_conventions is deprecated; use thurstone.conventions",
    DeprecationWarning,
    stacklevel=2,
)
