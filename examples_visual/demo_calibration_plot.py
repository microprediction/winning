from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, winner_of_many, sample_winner_of_many
from winning.lattice_calibration import solve_for_implied_offsets, ability_implied_state_prices, densities_from_offsets

PLOTS=True

unit = 0.1
L = 150

def demo():
    skew1                   = skew_normal_density(L=L, unit = unit, a=1.5)
    prices                  = [ 0.2, 0.3, 0.5 ]
    implied_offsets         = solve_for_implied_offsets(prices = prices, density = skew1, nIter = 2)
    inferred_prices         = ability_implied_state_prices( skew1, implied_offsets )
    print(str(inferred_prices))
    densities               = densities_from_offsets( skew1, implied_offsets )
    densityAllAgain, multiplicityAll  = winner_of_many(densities)
    if PLOTS:
        densitiesPlot( [ densityAllAgain ] + densities, unit = 0.1, legend = ['guess','analytic','1','2','3'] )


if __name__=='__main__':
    demo()