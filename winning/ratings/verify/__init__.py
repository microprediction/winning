"""winning.ratings.verify: the ratings layer's own verification suite.

    from winning.ratings.verify import verify
    report = verify(profile="fast")      # ok, or a list of what failed
    python -m winning.ratings.verify --profile full --workers 4 --out reports/

Profiles: smoke (seconds, runs on the installed wheel), fast (about a
minute, gates CI on every push), full (about half an hour, nightly and
on demand), exhaustive (no wall budget). Every check is deterministic
under marks.ROOT_SEED; marks live in marks.py and nowhere else. See
core.py for verdict semantics and research/adjudications/
ratings_verify.md for the adjudicated record.
"""

from .core import (CHECKS, PROFILES, VERDICTS, Check, Context,  # noqa: F401
                   Report, Result, check, seed_for, verify)
from . import marks  # noqa: F401
from . import identities  # noqa: F401  (registers checks)
from . import moments  # noqa: F401
from . import audit  # noqa: F401
from . import referee  # noqa: F401
