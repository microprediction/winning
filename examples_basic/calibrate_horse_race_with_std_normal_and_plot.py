from winning.std_calibration import centered_std_density
from winning.lattice_calibration import dividend_implied_ability
import numpy as np
import matplotlib.pyplot as plt
from winning.lattice_plot import densitiesPlot
from winning.lattice import skew_normal_density

# Illustrates the basic calibration
# Exactly the same but now we plot the densities


if __name__ =='__main__':

    # Choose the length of the lattice, which is 2*L+1
    L = 600

    # Choose the unit of discretization
    unit = 0.01

    # The unit is used to create an approximation of a density, here N(0,1) for simplicity
    density = centered_std_density(L=L, unit=unit)

    # Step 2. We set winning probabilities, most commonly represented in racing as inverse probabilities ('dividends')
    dividends = [2,6,np.nan, 3]

    # Step 3.  The algorithm implies relative ability (i.e. how much to translate the performance distributions)
    # Missing values will be assigned odds of 1999:1 ... or you can leave them out.
    abilities = dividend_implied_ability(dividends=dividends,density=density, nan_value=2000, unit=unit)


    densities = [skew_normal_density(L=L, unit=unit, loc=a, a=0, scale=1.0) for a in abilities]
    legend = [ str(d) for d in dividends ]
    densitiesPlot(densities=densities, unit=unit, legend=legend)
    plt.show()