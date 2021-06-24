from winning.std_calibration import centered_std_density
from winning.lattice_calibration import dividend_implied_ability
from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A
import numpy as np

# Illustrates the basic calibration
# Exactly the same but here we modify the discretization parameters


if __name__ =='__main__':

    # Choose the length of the lattice, which is 2*L+1
    L = 700

    # Choose the unit of discretization
    unit = 0.005

    # The unit is used to create an approximation of a density, here N(0,1) for simplicity
    density = centered_std_density(L=L, unit=unit)

    # Step 2. We set winning probabilities, most commonly represented in racing as inverse probabilities ('dividends')
    dividends = [2,6,np.nan, 3]

    # Step 3.  The algorithm implies relative ability (i.e. how much to translate the performance distributions)
    # Missing values will be assigned odds of 1999:1 ... or you can leave them out.
    abilities = dividend_implied_ability(dividends=dividends,density=density, nan_value=2000, unit=unit)

    # That's all. Lower ability is better.
    print(abilities)

    # Note that if you don't supply the unit, the abilities take on greater magnitudes than before (i.e. they are offsets on the lattice)
    # So you'll have to multiply them by the unit to get a scaled ability consistent with the density definition
    scale_free_abilities = dividend_implied_ability(dividends=dividends, density=density, nan_value=2000)
    scaled_ability = [ a*unit for a in scale_free_abilities ]
    print(scaled_ability)