from winning.lattice import fractional_shift
import numpy as np
from winning.lattice import _low_high, cdf_to_pdf, mean_of_density, fractional_shift_density
import numpy as np

def mean_of_cdf(cdf):
    pdf = cdf_to_pdf(cdf)
    return mean_of_density(pdf, unit=1)

def example_cdf():
    cdf = np.cumsum(np.random.rand(5))
    middle = cdf / cdf[-1]
    return np.concatenate([np.zeros(3),middle,np.ones(3)])

def test_shift_real():
    cdf = example_cdf()
    assert abs(np.max(cdf)-1)<1e-8
    x = 1.3
    cdf_moved = fractional_shift(cdf=cdf, x=x)
    mean_before = mean_of_cdf(cdf)
    mean_after = mean_of_cdf(cdf_moved)
    mean_diff = mean_after-mean_before
    assert abs(mean_diff-x)<1e-6


def test_shift_integer():
    cdf = example_cdf()
    assert abs(np.max(cdf)-1)<1e-8
    x = 1
    cdf_moved = fractional_shift(cdf=cdf, x=x)
    mean_before = mean_of_cdf(cdf)
    mean_after = mean_of_cdf(cdf_moved)
    mean_diff = mean_after-mean_before
    assert abs(mean_diff-x)<1e-6


def test_shift_pdf():
    cdf = example_cdf()
    pdf = cdf_to_pdf(cdf)
    x = 1.7
    pdf_moved = fractional_shift(cdf=pdf, x=x)
    mean_before = mean_of_density(pdf,unit=1)
    mean_after = mean_of_density(pdf_moved,unit=1)
    mean_diff = mean_after-mean_before
    assert abs(mean_diff-x)<1e-6


def test_shift_pdf_minus():
    cdf = example_cdf()
    pdf = cdf_to_pdf(cdf)
    x = -1.7
    pdf_moved = fractional_shift(cdf=pdf, x=x)
    mean_before = mean_of_density(pdf,unit=1)
    mean_after = mean_of_density(pdf_moved,unit=1)
    mean_diff = mean_after-mean_before
    assert abs(mean_diff-x)<1e-6


def test_shift_pdf_integer():
    cdf = example_cdf()
    pdf = cdf_to_pdf(cdf)
    x = 2
    pdf_moved = fractional_shift(cdf=pdf, x=x)
    mean_before = mean_of_density(pdf,unit=1)
    mean_after = mean_of_density(pdf_moved,unit=1)
    mean_diff = mean_after-mean_before
    assert abs(mean_diff-x)<1e-6
