import pytest

from winning.thurstone import AbilityCalibrator

# Skip if pytest-benchmark plugin is not installed
pytest.importorskip("pytest_benchmark")


def test_pricing_speed(benchmark, base):
    cal = AbilityCalibrator(base)
    ability = [0.0] * 24  # 24 runners, all similar

    def run():
        cal.state_prices_from_ability(ability)

    benchmark(run)
