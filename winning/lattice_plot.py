from winning.lattice import symmetric_lattice

def densitiesPlot( densities, unit, legend = None ):
    import matplotlib.pyplot as plt
    L = int(len( densities[0]-1 )/2)
    pts = symmetric_lattice( L=L, unit=unit )
    for density in densities:
        plt.plot( pts, density )
    if legend is not None:
        plt.legend( legend )
    plt.grid()
    plt.show()



