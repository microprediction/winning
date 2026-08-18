import numpy as np
import pytest

from winning.thurstone import Density, UniformLattice
from winning.thurstone.order_stats import expected_payoff_with_multiplicity, winner_of_many
from winning.thurstone.pricing import expected_payoff_vs_rest


def spike_density(lattice: UniformLattice, idx: int) -> Density:
    pdf = np.zeros(lattice.size, dtype=float)
    pdf[idx] = 1.0
    return Density(lattice, pdf)


def test_multiplicity_matches_number_of_tied_contestants():
    """
    On a coarse lattice, if we have k identical spike densities at the same lattice point,
    the multiplicity array returned by winner_of_many should be ≈ k at that score.
    """
    lattice = UniformLattice(L=3, unit=1.0)  # points: -3..+3
    center_idx = lattice.size // 2
    d = spike_density(lattice, center_idx)
    k = 4
    densities = [d] * k

    densityAll, multiplicityAll = winner_of_many(densities)
    m_center = float(multiplicityAll[center_idx])

    assert m_center == pytest.approx(k, rel=0.05), (
        f"Expected multiplicity ≈ {k} at tie point, got {m_center}"
    )
    # multiplicity shouldn't be < 1 anywhere
    assert np.all(multiplicityAll >= 1.0 - 1e-9)


def test_multiplicity_changes_draw_payoff_vs_flat_halfpoint():
    """
    In a dense tie scenario, the multiplicity-aware expected payoff
    should differ from a naive 'win + 0.5*draw' model.
    """
    lattice = UniformLattice(L=3, unit=1.0)
    center_idx = lattice.size // 2
    d = spike_density(lattice, center_idx)

    densities = [d, d, d, d]
    densityAll, multiplicityAll = winner_of_many(densities)

    payoff_mult = expected_payoff_with_multiplicity(d, densityAll, multiplicityAll)
    p_mult = float(np.sum(payoff_mult))

    all_min_cdf = densityAll.cdf()
    payoff_flat = expected_payoff_vs_rest(d, all_min_cdf)
    p_flat = float(np.sum(payoff_flat))

    assert not np.isclose(p_mult, p_flat, rtol=1e-6), (
        "Multiplicity-aware payoff should differ from naive half-point tie model"
    )
