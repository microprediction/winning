"""
Adaptive Special Horse Framework for Enhanced Thurstone Diffeomorphisms.

This module implements sophisticated special horse configurations with different
distributions, variances, and adaptive strategies to optimize mapping properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class DistributionType(Enum):
    """Supported distribution types for the special horse."""

    NORMAL = "normal"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


@dataclass
class SpecialHorseConfig:
    """Configuration for the special horse performance distribution."""

    distribution: DistributionType = DistributionType.NORMAL
    base_ability: float = 0.0  # Base ability level
    location: float = 0.0  # Mean/location parameter for noise
    scale: float = 1.0  # Standard deviation/scale parameter
    shape: float = 3.0  # Shape parameter (e.g., df for Student's t)
    adaptive_mode: str = "fixed"  # "fixed", "mean_adaptive", "position_adaptive"

    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("Scale parameter must be positive")
        if self.distribution == DistributionType.STUDENT_T and self.shape <= 0:
            raise ValueError("Student's t degrees of freedom must be positive")


class AdaptiveSpecialHorse:
    """
    Adaptive special horse that can use different distributions and strategies.
    """

    def __init__(self, config: SpecialHorseConfig):
        self.config = config
        self._rng = np.random.RandomState(42)  # For reproducibility

    def compute_ability(
        self,
        cube_point: np.ndarray,
        other_abilities: np.ndarray,
        sigmoid_params: Optional[Any] = None,
    ) -> float:
        """
        Compute the special horse ability based on configuration and context.

        Args:
            cube_point: Current position in cube [0,1]^k
            other_abilities: Abilities of the k regular horses
            sigmoid_params: Sigmoid parameters (for adaptive modes)

        Returns:
            Special horse ability
        """
        base_ability = self.config.base_ability

        # Adaptive ability adjustment
        if self.config.adaptive_mode == "mean_adaptive":
            # Set ability to balance the mean of other horses
            mean_other = np.mean(other_abilities)
            base_ability = -mean_other + self.config.base_ability

        elif self.config.adaptive_mode == "position_adaptive":
            # Vary ability based on cube position
            cube_center_distance = np.linalg.norm(cube_point - 0.5)
            base_ability = self.config.base_ability + 0.5 * cube_center_distance

        return base_ability

    def sample_performance(
        self,
        cube_point: np.ndarray,
        other_abilities: np.ndarray,
        sigmoid_params: Optional[Any] = None,
    ) -> float:
        """
        Sample a performance value for the special horse.

        Args:
            cube_point: Current position in cube
            other_abilities: Abilities of regular horses
            sigmoid_params: Sigmoid parameters (for adaptive modes)

        Returns:
            Sampled performance value
        """
        # Get base ability (potentially adaptive)
        ability = self.compute_ability(cube_point, other_abilities, sigmoid_params)

        # Sample noise from specified distribution
        noise = self._sample_noise()

        return ability + noise

    def _sample_noise(self) -> float:
        """Sample noise from the configured distribution."""

        if self.config.distribution == DistributionType.NORMAL:
            return self._rng.normal(self.config.location, self.config.scale)

        elif self.config.distribution == DistributionType.STUDENT_T:
            # Sample from Student's t distribution
            # We'll implement a simple approximation since scipy isn't available
            if self.config.shape > 30:
                # For large df, t-distribution approximates normal
                return self._rng.normal(self.config.location, self.config.scale)
            else:
                # Simple heavy-tailed approximation using mixture of normals
                if self._rng.random() < 0.1:  # 10% chance of heavy tail
                    scale_multiplier = 3.0
                else:
                    scale_multiplier = 1.0
                return self._rng.normal(self.config.location, self.config.scale * scale_multiplier)

        elif self.config.distribution == DistributionType.LAPLACE:
            # Laplace distribution using exponential random variables
            u = self._rng.random()
            if u < 0.5:
                return self.config.location - self.config.scale * np.log(2 * u)
            else:
                return self.config.location + self.config.scale * np.log(2 * (1 - u))

        elif self.config.distribution == DistributionType.UNIFORM:
            # Uniform distribution
            half_width = self.config.scale * np.sqrt(3)  # Match variance
            return self._rng.uniform(
                self.config.location - half_width, self.config.location + half_width
            )

        elif self.config.distribution == DistributionType.EXPONENTIAL:
            # Exponential distribution (only positive values)
            return self.config.location + self._rng.exponential(self.config.scale)

        else:
            raise ValueError(f"Unsupported distribution: {self.config.distribution}")

    def get_expected_performance(
        self,
        cube_point: np.ndarray,
        other_abilities: np.ndarray,
        sigmoid_params: Optional[Any] = None,
    ) -> float:
        """Get expected performance (for deterministic calculations)."""
        ability = self.compute_ability(cube_point, other_abilities, sigmoid_params)
        return ability + self.config.location

    def get_performance_variance(self) -> float:
        """Get variance of performance distribution."""
        if self.config.distribution == DistributionType.NORMAL:
            return self.config.scale**2
        elif self.config.distribution == DistributionType.STUDENT_T:
            if self.config.shape > 2:
                return self.config.scale**2 * self.config.shape / (self.config.shape - 2)
            else:
                return float("inf")  # Heavy tails
        elif self.config.distribution == DistributionType.LAPLACE:
            return 2 * (self.config.scale**2)
        elif self.config.distribution == DistributionType.UNIFORM:
            half_width = self.config.scale * np.sqrt(3)
            return (2 * half_width) ** 2 / 12
        elif self.config.distribution == DistributionType.EXPONENTIAL:
            return self.config.scale**2
        else:
            return self.config.scale**2  # Default fallback


def create_special_horse_configs() -> Dict[str, SpecialHorseConfig]:
    """Create a collection of interesting special horse configurations."""

    configs = {
        # Standard configurations
        "standard_normal": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
        ),
        # High variance for better coverage
        "high_variance": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.0,
            location=0.0,
            scale=2.0,
        ),
        # Low variance for smoothness
        "low_variance": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.0,
            location=0.0,
            scale=0.5,
        ),
        # Heavy tails for boundary coverage
        "heavy_tails": SpecialHorseConfig(
            distribution=DistributionType.STUDENT_T,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
            shape=3.0,  # Low df for heavy tails
        ),
        # Laplace for sharp peaked performance
        "laplace_peaked": SpecialHorseConfig(
            distribution=DistributionType.LAPLACE,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
        ),
        # Uniform for bounded performance
        "uniform_bounded": SpecialHorseConfig(
            distribution=DistributionType.UNIFORM,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
        ),
        # Mean-adaptive for automatic balancing
        "mean_adaptive": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
            adaptive_mode="mean_adaptive",
        ),
        # Position-adaptive for spatial variation
        "position_adaptive": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.0,
            location=0.0,
            scale=1.0,
            adaptive_mode="position_adaptive",
        ),
        # Strong special horse with low variance
        "dominant_reliable": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=1.0,
            location=0.0,
            scale=0.3,
        ),
        # Weak special horse with high variance
        "weak_unpredictable": SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=-0.5,
            location=0.0,
            scale=1.5,
        ),
    }

    return configs


# Example usage and testing
if __name__ == "__main__":
    print("ADAPTIVE SPECIAL HORSE FRAMEWORK TEST")
    print("=" * 50)

    # Create test configurations
    configs = create_special_horse_configs()

    # Test each configuration
    test_cube_point = np.array([0.3, 0.7])
    test_other_abilities = np.array([0.5, -0.2])

    for name, config in configs.items():
        print(f"\nTesting {name}:")
        print(f"   Distribution: {config.distribution.value}")
        print(f"   Base ability: {config.base_ability}")
        print(f"   Adaptive mode: {config.adaptive_mode}")

        special_horse = AdaptiveSpecialHorse(config)

        # Sample multiple performances
        performances = []
        for _ in range(1000):
            perf = special_horse.sample_performance(test_cube_point, test_other_abilities)
            performances.append(perf)

        performances = np.array(performances)

        print(f"   Mean performance: {np.mean(performances):.3f}")
        print(f"   Std performance: {np.std(performances):.3f}")
        print(f"   Performance range: [{np.min(performances):.3f}, {np.max(performances):.3f}]")

    print("\n✅ Adaptive special horse framework ready!")
    print("Ready to integrate with cube-to-simplex mappings!")
