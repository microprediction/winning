
from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, winner_of_many

unit = 0.05
L = 250


def demo():
    skew1 = skew_normal_density(L=L, unit = unit, a=1.5)
    skew2 = skew_normal_density(L=L, unit = unit, a=1.5, loc = -0.5)
    skew3 = skew_normal_density(L=L, unit = unit, a=1.5, loc = -1.0)
    best, multiplicity  = winner_of_many([skew1, skew2, skew3])
    densitiesPlot( [ skew1/unit, skew2/unit, skew3/unit, best /unit, multiplicity ], unit, legend=['1','2,','3','best', 'multiplicity'] )


if __name__=='__main__':
    demo()