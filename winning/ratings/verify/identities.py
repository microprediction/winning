"""Deterministic identity and metamorphic checks (I1-I8 of the plan).
Step 2 seeds the module with the sampler self-test; the moment
identities, invariances and cross-path checks follow in step 3."""

from __future__ import annotations

import numpy as np

from .core import Result, check


def _bases(which):
    from ...factor.races import (exponential_power_base, failure_base,
                                 skew_logistic_base, skew_normal_base,
                                 student_base)
    named = [("normal", "normal"), ("gumbel", "gumbel"),
             ("logistic", "logistic"), ("laplace", "laplace")]
    if which == "named":
        return named
    return named + [("student4", student_base(4.0)),
                    ("failure0.15", failure_base(0.15)),
                    ("failure_std", failure_base(0.15, standardize=True)),
                    ("expo1.5", exponential_power_base(1.5)),
                    ("expo6", exponential_power_base(6.0)),
                    ("skewlog0.5", skew_logistic_base(0.5)),
                    ("skewnorm2", skew_normal_base(2.0))]


@check("simulate.sampler_matches_density", profiles=("smoke",),
       group="simulate", cost_s=1.0)
def sampler_matches_density(ctx):
    """Every base's sampler against the base's own survival function:
    the Kolmogorov distance of n min-wins draws (winning.ratings.simulate
    .check_sampler). Exact samplers sit at the DKW floor; a sign or
    scale slip is two orders of magnitude above the mark."""
    from ..simulate import check_sampler
    n = int(ctx.param("ks_n"))
    m = ctx.mark()
    tol = m["tolerance_by_n"][n]
    out = []
    for name, base in _bases(ctx.param("ks_bases")):
        d = check_sampler(base, ctx.rng(name), n=n)
        out.append(Result(f"{ctx.name}.{name}", "simulate",
                          "ok" if d <= tol else "FAIL", statistic=d,
                          tolerance=tol, n=n, regime={"base": name},
                          detail="" if d <= tol else "sampler disagrees with its density"))
    return out
