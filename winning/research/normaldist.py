import math

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def normpdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def normcdf(x: float) -> float:
    # high-accuracy via erf
    return 0.5 * (1.0 + math.erf(x / SQRT2))
