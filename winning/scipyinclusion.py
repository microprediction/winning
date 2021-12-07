
try:
    from scipy.integrate import quad_vec
    using_scipy = True
except ImportError:
    using_scipy = False
