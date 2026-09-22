"""Tombstone: ``winning.thurstone`` was removed. Importing it raises an
ImportError that says where the code went and what to use instead.

It used to alias ``winning.research``, the retired research engine, and
the external ``thurstone`` package imported through it. A bare
``ModuleNotFoundError: No module named 'winning.thurstone'`` tells a
user nothing; this tells them the two things they need.
"""

raise ImportError(
    "winning.thurstone was removed.\n"
    "\n"
    "  The research engine it aliased is winning.research:\n"
    "      from thurstone import X            ->  from winning.research import X\n"
    "      from thurstone.inference import Y  ->  from winning.research.inference import Y\n"
    "\n"
    "  For calibration, prefer the front door, which is ~38x faster, "
    "round-trips to 1e-9 rather than 1e-5, and uses the compiled kernels:\n"
    "      winning.calibrate_abilities(p, V=, D=)\n"
    "\n"
    "  If this came from `import thurstone`, that package is retired: "
    "change its imports as above and drop the dependency."
)
