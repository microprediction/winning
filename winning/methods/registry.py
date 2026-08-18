"""Method registry: name -> callable(mu, V, D, budget, seed) -> (p, info)."""

METHODS = {}


def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def get_method(name):
    return METHODS[name]
