from winning.lattice import skew_normal_density
from winning.lattice_calibration import solve_for_implied_offsets, ability_implied_dividends
PLOTS=True
import math
import numpy as np

unit = 0.01
L = 500

def demo(  ):
    density = skew_normal_density(L=500, unit = unit, a=1.5)
    errors = list()
    race_sizes = [int(j) for j in np.logspace(base=10.,start=1,stop=5,num=20) ]

    for k,n in enumerate(race_sizes):
        print(n)
        print(errors[-1:])
        unnormalized_probs = np.linspace(start=5 / n, stop=5 / (n * math.log(n)), num=n)
        state_prices = [p_ / sum(unnormalized_probs) for p_ in unnormalized_probs]
        assert abs(sum(state_prices)-1)<1e-6
        implied_offsets = solve_for_implied_offsets(prices=state_prices, density=density, nIter=5)
        implied_dividends   = ability_implied_dividends(ability=implied_offsets,density=density)
        implied_state_prices = [1/dvd for dvd in implied_dividends ]
        relative_differences   = [ abs(p1-p2)/(p1) for p1, p2 in zip(state_prices,implied_state_prices)]
        avg_l1_error           = np.mean(np.abs( relative_differences ))
        errors.append( avg_l1_error)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(race_sizes[:k+1],errors)
    plt.xscale('log')
    plt.xlabel('Number of participants (n)')
    plt.ylabel('Mean relative error in win probability')
    plt.show()

if __name__=='__main__':
    demo()
