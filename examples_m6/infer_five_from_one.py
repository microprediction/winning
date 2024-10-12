from winning.lattice import skew_normal_density
import math
import matplotlib

if __name__=='__main__':
    from winning.lattice_plot import densitiesPlot
    a = 1.0  # Skew parameter
    densities = [skew_normal_density(L=100, unit=0.1, scale=1.0, loc=1.25*l-1.25/2, a=-1) for l in range(5)]
    densitiesPlot(densities, unit=0.1)
    matplotlib.pyplot.figure(num=None, figsize=None, dpi=3000, facecolor=None, edgecolor=None, frameon=True)

