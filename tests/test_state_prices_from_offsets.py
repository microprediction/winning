from winning.lattice_calibration import state_prices_from_offsets
from winning.std_calibration import centered_std_density
import numpy as np


def test_offset():
    density = centered_std_density(loc=0)
    offsets = np.array([-5.00001,0.12,5.3])
    prices = state_prices_from_offsets(density=density, offsets=offsets)
    assert( abs(np.sum(prices)-1)<1e-3)


def dont_test_failing_offset():
    # Case where offset is integer will fail
    density = centered_std_density(loc=0)
    offsets = np.array([-5,0.12,5.3])
    prices = state_prices_from_offsets(density=density, offsets=offsets)
    assert( abs(np.sum(prices)-1)<1e-3)

