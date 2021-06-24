from winning.std_calibration import centered_std_density
from winning.lattice_calibration import dividend_implied_ability
import numpy as np

# Illustrates the basic calibration


if __name__ =='__main__':

    # Step 1. We choose a performance density
    density = centered_std_density()

    # Step 2. We set winning probabilities, most commonly represented in racing as inverse probabilities ('dividends')
    dividends = [2,6,np.nan, 3]

    # Step 3.  The algorithm implies relative ability (i.e. how much to translate the performance distributions)
    # Missing values will be assigned odds of 1999:1 ... or you can leave them out.
    abilities = dividend_implied_ability(dividends=dividends,density=density, nan_value=2000)

    # That's all. Lower ability is better.
    print(abilities)

    # Because this is so common there's a one-liner alternative
    from winning.std_calibration import std_dividend_implied_ability
    abilities1 = std_dividend_implied_ability(dividends=dividends, nan_value=2000)


