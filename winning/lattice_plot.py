from winning.lattice import symmetric_lattice

try:
    import matplotlib.pyplot as plt
    using_plots = True
except:
    using_plots = False
    print('pip install matplotlib')


def densitiesPlot( densities, unit, legend = None ):
    assert using_plots
    L = int(len( densities[0]-1 )/2)
    pts = symmetric_lattice( L=L, unit=unit )
    for density in densities:
        plt.plot( pts, density )
    if legend is not None:
        plt.legend( legend )
    plt.grid()
    plt.show()


def std_plot( abilities, legend = None):
    from winning.std_calibration import centered_std_density
    densities = [centered_std_density(loc=a) for a in abilities]
    densitiesPlot(densities=densities, legend=legend)
