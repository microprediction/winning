from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, mean_of_density, implicit_state_prices, winner_of_many, sample_winner_of_many
from winning.lattice_calibration import implied_ability, state_prices_from_offsets, densities_from_offsets

PLOTS=True

unit = 0.1
L = 150

def demo():
    skew1                   = skew_normal_density(L=L, unit = unit, a=1.5)
    prices                  = [ 0.2, 0.3, 0.5 ]
    implied_offsets         = implied_ability(prices = prices, density = skew1, nIter = 2)
    inferred_prices         = state_prices_from_offsets( skew1, implied_offsets )
    print(str(inferred_prices))
    densities               = densities_from_offsets( skew1, implied_offsets )
    densityAllAgain, multiplicityAll  = winner_of_many(densities)
    if PLOTS:
        densitiesPlot( [ densityAllAgain ] + densities, unit = 0.1, legend = ['guess','analytic','1','2','3'] )


if __name__=='__main__':
    demo()