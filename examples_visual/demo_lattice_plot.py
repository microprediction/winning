from winning.lattice_plot import  densitiesPlot
from winning.lattice import skew_normal_density, mean_of_density

PLOTS=True

unit = 0.1
L = 150


def demo(  ):
    skew = skew_normal_density(L=L, unit = unit, a=1.0)
    print("mean is " + str(mean_of_density(skew, unit=unit)))
    densitiesPlot( [skew], unit=unit )


if __name__=='__main__':
    demo()