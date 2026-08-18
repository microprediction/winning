# Gets rid of scipy dependency which is a pain on M1
# These are never used in the inner loop



def backward_compat_normcdf_function():
    try:
        from statistics import NormalDist
        return NormalDist(mu=0, sigma=1.0).cdf
    except ImportError:
        try:
            from scipy.stats import norm
            return norm.cdf
        except ImportError:
            raise Exception('You need to install scipy or a version of Python with statistics.NormalDist')

normcdf = backward_compat_normcdf_function()


def backward_compat_invnormcdf_function():
    try:
        from statistics import NormalDist
        return NormalDist(mu=0, sigma=1.0).inv_cdf
    except ImportError:
        try:
            from scipy.stats import norm
            return norm.ppf
        except ImportError:
            raise Exception('You need to install scipy or a version of Python with statistics.NormalDist')

invnormcdf = backward_compat_invnormcdf_function()


def backward_compat_normpdf_function():
    try:
        from statistics import NormalDist
        return NormalDist(mu=0, sigma=1.0).pdf
    except ImportError:
        try:
            from scipy.stats import norm
            return norm.pdf
        except ImportError:
            raise Exception('You need to install scipy or a version of Python with statistics.NormalDist')


normpdf = backward_compat_normpdf_function()

