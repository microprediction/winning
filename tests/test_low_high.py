
from winning.lattice import _low_high, cdf_to_pdf
import numpy as np


def cdf_mean(cdf):
   pdf = cdf_to_pdf(cdf)
   v = np.cumsum(np.ones(len(pdf)))
   return np.dot(pdf,v)


def test_low_high():
   l,h =  _low_high(offset=0.7, L=10)
   assert l[0]==0
   assert h[0]==1
   assert abs(l[1] - 0.3) < 1e-6
   assert abs(h[1]-0.7)<1e-6


def test_low_high():
   l,h =  _low_high(offset=2.0, L=10)
   assert l[0]==2
   assert h[0]==2
   assert abs(l[1]+h[1] - 1.0) < 1e-6


def test_low_high():
   l,h =  _low_high(offset=-0.3, L=10)
   assert l[0]==-1
   assert h[0]==0
   assert abs(l[1] - 0.3) < 1e-6
   assert abs(h[1]-0.7)<1e-6