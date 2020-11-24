from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, winner_of_many, sample_winner_of_many

PLOTS=True

unit = 0.1
L = 150

def demo():
    skew1     = skew_normal_density(L=25, unit = unit, a=1.5)
    densities = [ skew1, skew1, skew1 ]
    densityAll, multiplicityAll  = winner_of_many(densities)
    densityAllCheck = sample_winner_of_many(densities, nSamples = 50000)
    if PLOTS:
        densitiesPlot( [ densityAll, densityAllCheck ], unit = 0.1 )


if __name__=='__main__':
    demo()