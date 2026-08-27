"""Cross-repo behavioral contract: seeded golden checks over the core
APIs, runnable by any consumer.

The problem this solves: the winning core is now consumed by several
repositories and two vendored ports; version pins protect against API
breakage but not behavior drift. Downstream repos add one test that
calls winning.contract.verify() so their CI fails the moment an
installed winning behaves differently from the recorded contract.

    python -m winning.contract        # or
    from winning.contract import verify; verify()

Golden values are computed from seeded inputs and asserted to stated
tolerances; regenerating them is a deliberate act (edit this file),
never a side effect.
"""

from __future__ import annotations

import numpy as np


def verify(verbose=True):
    failures = []

    def check(name, value, expect, tol):
        ok = abs(value - expect) <= tol
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
                  f"{value:.12g} vs {expect:.12g} (tol {tol:g})")
        if not ok:
            failures.append(name)

    from winning.factor.races import race_probabilities
    rng = np.random.default_rng(4)
    mu = rng.normal(size=10)
    V = rng.normal(size=(10, 2)) * 0.4
    D = 0.5 + rng.random(10)

    p = race_probabilities(mu, V=V, D=D, points=257)
    check("factor race p[0]", float(p[0]), 0.12853275, 5e-7)
    check("factor race sum", float(p.sum()), 1.0, 1e-12)

    c = np.pi / np.sqrt(6.0)
    soft = np.exp(-c * mu); soft = soft / soft.sum()
    pg = race_probabilities(mu, base="gumbel", points=2001)
    check("gumbel = softmax, max rel",
          float(np.max(np.abs(pg - soft) / soft)), 0.0, 1e-12)

    from winning.factor.core import (abilities_from_probabilities_factor,
                                     hermite_nodes)
    F2, W2 = hermite_nodes(2, 7)
    p_t = np.array([0.4, 0.3, 0.2, 0.05, 0.03, 0.02])
    m = abilities_from_probabilities_factor(p_t, V[:6], D[:6], F2, W2,
                                            points=513, tol=1e-9)
    p_b = race_probabilities(m, V=V[:6], D=D[:6], F=F2, W=W2, points=1025)
    check("inversion round trip, max |log ratio|",
          float(np.max(np.abs(np.log(p_b / p_t)))), 0.0, 1e-6)

    from winning.likelihood import choice_loglik_and_score
    rng2 = np.random.default_rng(2)
    mu2 = rng2.normal(size=(20, 4))
    V2 = rng2.normal(size=(4, 2)) * 0.5
    ch = rng2.integers(0, 4, 20)
    ll, dmu, dV = choice_loglik_and_score(mu2, V2, ch)
    check("likelihood value", float(ll), -51.33284124278608, 1e-6)
    check("likelihood common-shift invariance",
          float(choice_loglik_and_score(mu2 + 2.5, V2, ch)[0] - ll),
          0.0, 1e-9)

    from winning.fastmvn import mvn_cdf_fast
    from scipy.stats import norm
    D6 = 0.5 + np.arange(1, 7) / 10
    b6 = np.linspace(-1, 1.5, 6)
    check("fastmvn independence identity",
          float(mvn_cdf_fast(upper=b6, V=np.zeros((6, 1)), D=D6)),
          float(np.prod(norm.cdf(b6 / np.sqrt(D6)))), 1e-12)

    if failures:
        raise AssertionError(f"winning contract violated: {failures}")
    if verbose:
        print("winning contract: all checks pass")
    return True


if __name__ == "__main__":
    verify()
