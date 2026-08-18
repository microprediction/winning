from winning.lattice_simulation import skew_normal_place_pricing


def test_pricing_symmetric():
    dividends = [6, 6, 6, 6, 6, 6]
    prices = skew_normal_place_pricing(dividends=dividends, longshot_expon=1.17, n_samples=500)
    assert(all([ abs(p-2.0)<0.5 for p in prices['place3']] ))

