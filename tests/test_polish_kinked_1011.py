"""#1011: polish_race on a kinked base polishes on the kink-split lattice.

The uniform lattice's window slides with mu across every laplace
runner's kink, so the map's derivative carries an O(dx) artifact that
race_jacobian (kink_split=False) does not match: 1.2e-2 off the split
Jacobian on the issue's three-runner race, with rows summing to 1e-2.
SLSQP on that pair hit its iteration limit, converged=False. polish_race
now runs forward, Jacobian and inverse on the split lattice for a base
that declares kinks; a smooth base is untouched, and the default forward
race_probabilities is unchanged for every base.
"""
import numpy as np
import pytest

from winning.factor.polish import polish_race, race_jacobian
from winning.factor.races import race_probabilities, exponential_power_base


def _cases():
    rng = np.random.default_rng(0)
    out = []
    for n in (10, 6, 15, 4):
        out.append((rng.normal(size=n) * 0.8, rng.uniform(0.5, 1.5, n),
                    1.5 / n + 0.05))
    return out


def test_issue_jacobian_gap_is_the_uniform_lattice_artifact():
    # the documented defect, pinned so the docstring stays honest: the
    # default Jacobian is first-order off, the split one is not
    mu = np.array([0.0, 0.4, -0.3])
    D = np.array([1.0, 0.7, 1.3])
    J = race_jacobian(mu, D=D, base="laplace")
    Js = race_jacobian(mu, D=D, base="laplace", kink_split=True)
    assert np.abs(J - Js).max() > 1e-3
    assert np.abs(Js.sum(axis=1)).max() < 1e-12
    h = 1e-6
    fd = np.column_stack([
        (race_probabilities(mu + h * e, D=D, base="laplace", points=501,
                            kink_split=True)
         - race_probabilities(mu - h * e, D=D, base="laplace", points=501,
                              kink_split=True)) / (2 * h)
        for e in np.eye(3)])
    assert np.abs(Js - fd).max() < 1e-8


@pytest.mark.parametrize("base", ["laplace", exponential_power_base(1.2)])
def test_kinked_polish_converges(base):
    for mu0, D, cap in _cases():
        p, mu, info = polish_race(mu0=mu0, D=D, base=base, name_caps=cap)
        assert info["kink_split"] is True
        assert info["converged"], (len(mu0), info)
        assert info["nit"] < 20
        assert p.max() <= cap + 1e-6
        # the returned p IS the split-lattice race at mu
        np.testing.assert_array_equal(
            p, race_probabilities(mu, D=D, base=base, kink_split=True))


def test_kinked_polish_on_the_uniform_lattice_is_opt_in():
    mu0, D, cap = _cases()[0]
    p, mu, info = polish_race(mu0=mu0, D=D, base="laplace", name_caps=cap,
                              kink_split=False)
    assert info["kink_split"] is False
    np.testing.assert_array_equal(
        p, race_probabilities(mu, D=D, base="laplace"))


def test_kinked_p0_round_trips_through_the_split_inverse():
    mu = np.array([0.0, 0.4, -0.3, 0.2])
    D = np.array([1.0, 0.7, 1.3, 0.9])
    V = np.array([[0.3], [-0.2], [0.5], [0.0]])
    p0 = race_probabilities(mu, V=V, D=D, base="laplace", kink_split=True)
    # a slack cap: the polished race is p0 itself
    p, _, info = polish_race(p0=p0, V=V, D=D, base="laplace",
                             A=np.array([[1.0, 0, 0, 0]]), b=np.array([0.99]))
    assert info["converged"]
    assert np.abs(p - p0).max() < 1e-8
    p, _, info = polish_race(p0=p0, V=V, D=D, base="laplace",
                             name_caps=p0.max() - 0.02)
    assert info["converged"] and info["max_violation"] == 0.0


@pytest.mark.parametrize("base", ["normal", "logistic", "gumbel"])
def test_smooth_polish_is_unchanged(base):
    mu0, D, cap = _cases()[0]
    a = polish_race(mu0=mu0, D=D, base=base, name_caps=cap)
    b = polish_race(mu0=mu0, D=D, base=base, name_caps=cap, kink_split=False)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2]["kink_split"] is False and a[2]["converged"]


def test_kink_split_door():
    from winning.factor.structures import Factor
    with pytest.raises(ValueError, match="structure"):
        polish_race(mu0=np.zeros(3), structure=Factor(np.zeros((3, 1)),
                                                      np.ones(3)),
                    name_caps=0.5, kink_split=True)
    with pytest.raises(TypeError, match="kink_split"):
        polish_race(mu0=np.zeros(3), D=np.ones(3), base="laplace",
                    name_caps=0.5, kink_split="yes")
