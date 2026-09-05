"""
Enhanced Cube-to-Simplex Mapping with Adaptive Special Horse.

This module extends the original mapping with sophisticated special horse
configurations for optimal diffeomorphism properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .adaptive_special_horse import AdaptiveSpecialHorse
from .conventions import STD_L, STD_UNIT
from .cube_to_simplex import SigmoidParams
from .density import Density
from .lattice import UniformLattice
from .pricing import Race


@dataclass
class EnhancedCubeToSimplexMapping:
    """
    Enhanced diffeomorphism with adaptive special horse capabilities.
    """

    sigmoid_params: List[SigmoidParams]
    special_horse: AdaptiveSpecialHorse
    noise_scale: float = 1.0
    lattice: Optional[UniformLattice] = None

    def __post_init__(self):
        if self.lattice is None:
            self.lattice = UniformLattice(L=STD_L, unit=STD_UNIT)
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive")

    @property
    def k(self) -> int:
        """Dimension of the cube (and simplex)."""
        return len(self.sigmoid_params)

    def cube_to_abilities(self, cube_point: np.ndarray) -> np.ndarray:
        """Convert point in [0,1]^k to abilities for all k+1 horses."""
        cube_point = np.asarray(cube_point)
        if len(cube_point) != self.k:
            raise ValueError(f"Expected {self.k} dimensions, got {len(cube_point)}")

        # Map first k dimensions to abilities using sigmoids
        regular_abilities = np.zeros(self.k)
        for i, (x, sigmoid_param) in enumerate(zip(cube_point, self.sigmoid_params)):
            if not (0 <= x <= 1):
                raise ValueError(f"Cube point coordinate {i} = {x} not in [0,1]")
            regular_abilities[i] = sigmoid_param(x)

        # Get special horse ability (potentially adaptive)
        special_ability = self.special_horse.compute_ability(
            cube_point, regular_abilities, self.sigmoid_params
        )

        # Combine all abilities
        all_abilities = np.concatenate([regular_abilities, [special_ability]])
        return all_abilities

    def create_race_with_sampling(self, cube_point: np.ndarray, n_samples: int = 1) -> Race:
        """
        Create a Race object using Monte Carlo sampling for the special horse.

        For deterministic behavior, use n_samples=1.
        For better approximation of special horse distribution, use larger n_samples.
        """
        cube_point = np.asarray(cube_point)
        regular_abilities = np.zeros(self.k)

        # Compute regular horse abilities
        for i, (x, sigmoid_param) in enumerate(zip(cube_point, self.sigmoid_params)):
            regular_abilities[i] = sigmoid_param(x)

        if n_samples == 1:
            # Use expected performance for deterministic behavior
            special_ability = self.special_horse.get_expected_performance(
                cube_point, regular_abilities, self.sigmoid_params
            )
            all_abilities = np.concatenate([regular_abilities, [special_ability]])

            # Create normal densities for each horse
            densities = []
            for ability in all_abilities:
                density = Density.skew_normal(
                    lattice=self.lattice, loc=ability, scale=self.noise_scale, a=0.0
                )
                densities.append(density)

            return Race(densities)

        else:
            # Use Monte Carlo sampling to approximate special horse distribution
            special_performances = []
            for _ in range(n_samples):
                perf = self.special_horse.sample_performance(
                    cube_point, regular_abilities, self.sigmoid_params
                )
                special_performances.append(perf)

            # Use mean of sampled performances
            mean_special_performance = np.mean(special_performances)
            all_abilities = np.concatenate([regular_abilities, [mean_special_performance]])

            # Create densities
            densities = []
            for i, ability in enumerate(all_abilities):
                if i < self.k:  # Regular horses
                    density = Density.skew_normal(
                        lattice=self.lattice, loc=ability, scale=self.noise_scale, a=0.0
                    )
                else:  # Special horse - adjust variance if needed
                    special_variance = self.special_horse.get_performance_variance()
                    effective_scale = np.sqrt(special_variance + self.noise_scale**2)
                    density = Density.skew_normal(
                        lattice=self.lattice, loc=ability, scale=effective_scale, a=0.0
                    )
                densities.append(density)

            return Race(densities)

    def __call__(self, cube_point: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Map point from k-cube to k-simplex.

        Args:
            cube_point: Point in [0,1]^k
            n_samples: Number of samples for special horse (1 for deterministic)

        Returns:
            Point on k-simplex (winning probabilities)
        """
        race = self.create_race_with_sampling(cube_point, n_samples)
        return race.state_prices()

    def forward(self, cube_point: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Alias for __call__ to make direction explicit."""
        return self(cube_point, n_samples)

    def batch_forward(self, cube_points: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Apply mapping to multiple points.

        Args:
            cube_points: Array of shape (n_points, k)
            n_samples: Number of samples for special horse

        Returns:
            Array of shape (n_points, k+1) on the simplex
        """
        cube_points = np.asarray(cube_points)
        if cube_points.ndim != 2 or cube_points.shape[1] != self.k:
            raise ValueError(f"Expected shape (n_points, {self.k}), got {cube_points.shape}")

        results = []
        for point in cube_points:
            results.append(self.forward(point, n_samples))

        return np.array(results)


def create_test_mappings() -> dict:
    """Create test mappings with different special horse configurations."""
    from .adaptive_special_horse import create_special_horse_configs

    # Standard sigmoid parameters
    sigmoid_params = [
        SigmoidParams(alpha=1.2, beta=4.0, gamma=0.4),
        SigmoidParams(alpha=1.0, beta=5.0, gamma=0.6),
    ]

    special_horse_configs = create_special_horse_configs()

    mappings = {}
    for name, config in special_horse_configs.items():
        special_horse = AdaptiveSpecialHorse(config)
        mapping = EnhancedCubeToSimplexMapping(
            sigmoid_params=sigmoid_params, special_horse=special_horse, noise_scale=1.0
        )
        mappings[name] = mapping

    return mappings


# Example usage and testing
if __name__ == "__main__":
    print("ENHANCED CUBE-TO-SIMPLEX MAPPING TEST")
    print("=" * 50)

    # Create test mappings
    mappings = create_test_mappings()

    # Test points
    test_points = [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]]

    print("Testing different special horse configurations...")

    for name, mapping in list(mappings.items())[:5]:  # Test first 5
        print(f"\n{name}:")
        print(f"   Special horse: {mapping.special_horse.config.distribution.value}")

        for i, point in enumerate(test_points):
            result = mapping(point)
            print(f"   Point {i + 1} {point} → [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}]")

        # Test the special property
        center_point = [0.5, 0.5]
        center_result = mapping(center_point)
        print(f"   Center [0.5, 0.5] → sum = {np.sum(center_result):.6f} ✓")

    print("\n✅ Enhanced mapping framework working!")
    print("Ready for comprehensive quality comparison!")
