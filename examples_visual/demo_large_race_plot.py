from winning.lattice_plot import densitiesPlot
from winning.lattice import skew_normal_density, winner_of_many

unit = 0.05
L = 250


def demo():
    densities = [skew_normal_density(L=L, unit=unit, a=0.3 * i) for i in range(25)]
    best, multiplicity = winner_of_many(densities)
    densitiesPlot([d / unit for d in densities[:5]] + [best / unit, multiplicity], unit)

if __name__=='__main__':
     demo()