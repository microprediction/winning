"""Finite dividends whose reciprocals overflow are an ordinary relative book (#881).

[1e-320, 2e-320] took 1/x to [inf, inf] and normalised to inf/inf = NaN,
which the classic inverse then refused. The book is read against its
smallest positive dividend instead, in python, the browser and R.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density
from winning.classic.lattice_calibration import dividend_implied_ability, prices_from_dividends

NODE = shutil.which("node")
CASES = [
    ([1e-320, 2e-320], [2 / 3, 1 / 3]),
    ([1e-320, 1e-320, 4e-320], [4 / 9, 4 / 9, 1 / 9]),
    ([1e-320, 0.0, -1.0, float("inf")], [1.0, 0.0, 0.0, 0.0]),
    ([1e-320, 1e10], [1.0, 1e-330]),          # the long shot underflows to 0
]


@pytest.mark.parametrize("dividends, want", CASES)
def test_overflowing_reciprocals_price_the_relative_book(dividends, want):
    got = prices_from_dividends(dividends)
    assert np.all(np.isfinite(got)) and abs(sum(got) - 1) < 1e-15
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300)


def test_ordinary_books_are_bit_identical():
    d = [2.5, 3.3, 5.0, 10.0, float("nan"), 0.0]
    inv = [1 / 2.5, 1 / 3.3, 1 / 5.0, 1 / 10.0, 1 / 2000.0, 0.0]
    assert prices_from_dividends(d) == [x / sum(inv) for x in inv]


def test_the_inverse_accepts_the_book():
    mu = dividend_implied_ability([1e-320, 2e-320], skew_normal_density(L=15, unit=0.1, a=0))
    assert np.all(np.isfinite(mu)) and mu[0] < mu[1]   # the shorter price is the favourite


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_overflowing_reciprocals_price_the_relative_book():
    script = f"""
import {{ pricesFromDividends }} from "./docs/js/winning/classic.mjs";
console.log(JSON.stringify({json.dumps([c[0] for c in CASES])}
  .map(d => pricesFromDividends(d))));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                         text=True, check=True)
    for got, (_, want) in zip(json.loads(out.stdout), CASES):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300)
