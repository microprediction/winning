from winning.lattice import density_from_samples, mean_of_density
import numpy as np


def test_samples():
    x = np.random.randn(5000)
    unit = 0.05
    d = density_from_samples(x=x,L=501, unit=unit)
    mu = mean_of_density(density=d,unit=unit)



if __name__=='__main__':
    x = np.random.randn(5000)
    unit = 0.05
    d = density_from_samples(x=x, L=501, unit=unit)
    mu = mean_of_density(density=d, unit=unit)
    print(mu / unit)
    import matplotlib.pyplot as plt

    plt.plot(d)
    plt.show()