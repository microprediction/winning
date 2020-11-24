from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, mean_of_density, implicit_state_prices, winner_of_many, sample_winner_of_many
from winning.lattice_calibration import implied_ability, state_prices_from_offsets, densities_from_offsets
import numpy as np
PLOTS=True
import math

unit = 0.05
L = 500

def demo(  ):
    density                 = skew_normal_density(L=500, unit = unit, a=1.5)
    n                       = 100

    errors = list()
    num_iters = list(range(1,5))
    for num_iter in num_iters:
        true_offsets            = [ int(unit*k) for k in range( n ) ]
        state_prices            = state_prices_from_offsets( density=density, offsets=true_offsets )
        print("State prices are " + str( state_prices ))
        offset_samples          = list( range( -100, 100 ))[::-1]
        # Now try to infer offsets from state prices
        implied_offsets         = implied_ability( prices = state_prices, density = density, offset_samples= offset_samples, nIter=num_iter)
        recentered_offsets      = [ io-implied_offsets[0] for io in implied_offsets]
        differences             = [ o1-o2 for o1, o2 in zip(recentered_offsets,true_offsets)]
        avg_l1_in_offset        = np.mean(np.abs( differences ))
        errors.append( avg_l1_in_offset)
        print(avg_l1_in_offset)

    import matplotlib.pyplot as plt
    plt.scatter(num_iters,errors)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean absolute error in relative ratings')
    plt.show()

if __name__=='__main__':
    demo()
