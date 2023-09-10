from winning.lattice import density_from_samples, mean_of_density
import numpy as np


def test_samples():
    x = np.random.randn(5000)
    unit = 0.05
    d = density_from_samples(x=x,L=501, unit=unit)
    mu = mean_of_density(density=d,unit=unit)


def do_die():
    x = [ np.random.choice([-2.5,-1.5,-0.5,0, 0.5,1.5,2.5]) for _ in range(10000) ]
    unit = 1
    d = density_from_samples(x=x,L=5, unit=0.5)
    mu = mean_of_density(density=d,unit=unit)
    print(mu)
    return d


def test_die():
    do_die()


if __name__=='__main__':
    d = do_die()
    import matplotlib.pyplot as plt
    print(d)

    plt.plot(d)
    plt.show()