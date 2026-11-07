"""A floor above 1/n from a saturated start reaches the binding point (#840).

The uniform race is infeasible under p[0] >= 0.9, so the uniform-ray
restoration never ran: the browser crossed to the opposite deterministic
winner and certified it, and python returned the violating start
unconverged. Both now walk from the uniform race along the violated rows'
direction to a feasible start.
"""
import numpy as np
import pytest

from winning.factor.polish import polish_race

X_STAR = 0.9061938024368232            # -Phi^-1(0.9) / sqrt(2)


@pytest.mark.parametrize("a", [8 / np.sqrt(2), 4.9, 2.0])
def test_floored_pair_reaches_the_binding_point(a):
    p, mu, info = polish_race(mu0=[a, -a], D=[1, 1], A=[[-1, 0]], b=[-0.9])
    assert info["converged"] and list(info["active"]) == [0]
    np.testing.assert_allclose(p, [0.9, 0.1], atol=1e-8)
    np.testing.assert_allclose(mu, [-X_STAR, X_STAR], atol=1e-7)
    assert info["mu_distance"] == pytest.approx(np.sqrt(2) * (a + X_STAR), abs=1e-6)
