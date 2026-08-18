import numpy as np
from numpy.testing import assert_allclose

from winning.thurstone import AbilityCalibrator, Density


def draw_from_density(d: Density, rng, n=50_000):
    cdf = np.cumsum(d.p)
    u = rng.random(n)
    # integer sample (index), convert to physical
    idx = np.searchsorted(cdf, u)
    xs = d.lattice.unit * (idx - d.lattice.L)
    return xs


def test_mc_vs_lattice(base, rng):
    # 3 runners with modest offsets
    ability = [-1.0, 0.0, +0.5]
    cal = AbilityCalibrator(base)
    p_lattice = cal.state_prices_from_ability(ability)

    # MC simulate min
    dens = [base.shift_fractional(a / base.lattice.unit) for a in ability]
    samples = [draw_from_density(d, rng) for d in dens]
    N = len(samples[0])
    wins = [0, 0, 0]
    for k in range(N):
        # lower wins
        vals = [samples[i][k] for i in range(3)]
        w = int(np.argmin(vals))
        wins[w] += 1
    p_mc = [w / N for w in wins]
    assert_allclose(p_mc, p_lattice, atol=0.02)  # coarse tolerance is fine
