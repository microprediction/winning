
try:
    from scipy.integrate import quad_vec
    using_scipy = True
except ImportError:
    using_scipy = False

if __name__=='__main__':
    print({'using_scipy':using_scipy})