from winning.lattice_pricing import skew_normal_simulation

def test_pricing_symmetric():
    dividends = [6, 6, 6, 6, 6, 6]
    prices = skew_normal_simulation(dividends=dividends, longshot_expon=1.17, nSamples=500)
    assert(all([ abs(p-2.0)<0.5 for p in prices['show']] ))

