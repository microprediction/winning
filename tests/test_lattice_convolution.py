from winning.lattice import convolve_two, mean_of_density, convolve_many, skew_normal_density


def test_two():
    density1 = [0, 0, 0, 0, 0.1, 0.4,   0.2, 0.1,  0.1,  0.1, 0, 0, 0, 0, 0]
    density2 =    [0, 0, 0, 0.1, 0.15, 0.35, 0.15, 0.15, 0.1, 0, 0, 0, 0]
    density = convolve_two(density1=density1, density2=density2, L=20)
    mu_before = mean_of_density(density1, unit=1)+mean_of_density(density2, unit=1)
    mu_after = mean_of_density(density, unit=1)
    assert abs(mu_after-mu_before)<1e-4


def test_two_skew():
    density1 = skew_normal_density(L=50, unit=1.0, scale=5.0, a=1.0)
    density2 = skew_normal_density(L=50, unit=1.0, scale=5.0, loc=1.2, a=-1.0)
    density = convolve_two(density1=density1, density2=density2, L=100)
    mu_before = mean_of_density(density1, unit=1)+mean_of_density(density2, unit=1)
    mu_after = mean_of_density(density, unit=1)
    assert abs(mu_after-mu_before)<1e-4


def test_many_skew():
    densities = [ skew_normal_density(L=50, unit=1.0, scale=5.0, a=1.0) for _ in range(10) ]
    density = convolve_many(densities=densities, L=500, do_padding=True)
    assert len(density)==1001
    mu_before = sum( [ mean_of_density(d, unit=1) for d in densities ] )
    mu_after = mean_of_density(density, unit=1)
    assert abs(mu_after-mu_before)<1e-4


def test_many_many_normal():
    densities = [ skew_normal_density(L=50, unit=1.0, scale=5.0, a=0.0) for _ in range(100) ]
    L = 5000
    density = convolve_many(densities=densities, L=L)
    assert len(density)==2*L+1
    mu_before = sum( [ mean_of_density(d, unit=1) for d in densities ] )
    mu_after = mean_of_density(density, unit=1)
    assert abs(mu_after-mu_before)<1e-4




if __name__=='__main__':
    test_two()
    test_many_skew()
    test_many_many_normal()
    test_two_skew()