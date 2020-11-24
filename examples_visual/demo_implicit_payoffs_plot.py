import winning.lattice_plot
from winning.lattice import skew_normal_density, mean_of_density, implicit_state_prices, winner_of_many

PLOTS=True

unit = 0.1
L = 150

def demo(  ):
    skew1 = skew_normal_density(L=L, unit = unit, a=1.5)
    skew2 = skew_normal_density(L=L, unit = unit, a=1.5, loc = -0.5)
    skew3 = skew_normal_density(L=L, unit = unit, a=1.5, loc = -1.0)
    densityAll, multiplicityAll  = winner_of_many([skew1, skew2, skew3])
    payoffs = implicit_state_prices( density=skew1, densityAll=densityAll, multiplicityAll = multiplicityAll, cdf = None, cdfAll = None, offsets = None )
    import matplotlib.pyplot as plt
    plt.plot( payoffs )
    plt.show()

if __name__=='__main__':
    demo()