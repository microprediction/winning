from winning.lattice import skew_normal_density


if __name__=='__main__':
    from winning.lattice_plot import densitiesPlot
    a = 1.0  # Skew parameter
    densities = [skew_normal_density(L=100, unit=0.1, scale=1.0, loc=0, a=a) for _ in range(5)]
    from winning.lattice_calibration import state_price_implied_ability
    abilities = state_price_implied_ability(densities)