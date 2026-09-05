"""
Diffeomorphisms from k-cube to k-simplex using existing Thurstone infrastructure.

This module builds on the existing Race and Density classes to create
smooth mappings from [0,1]^k to the k-simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .conventions import STD_L, STD_UNIT
from .density import Density
from .lattice import UniformLattice
from .pricing import Race


def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow


def parametric_sigmoid(x: float, alpha: float, beta: float, gamma: float) -> float:
    """
    Parametric sigmoid: f(x) = alpha * sigmoid(beta * (x - gamma))

    Args:
        x: Input in [0,1]
        alpha: Scale parameter
        beta: Steepness parameter (positive for increasing)
        gamma: Shift parameter (inflection point)

    Returns:
        Mapped value (typically ability)
    """
    return alpha * sigmoid(beta * (x - gamma))


@dataclass
class SigmoidParams:
    """Parameters for a single sigmoid mapping."""

    alpha: float  # Scale
    beta: float  # Steepness
    gamma: float  # Shift (inflection point)

    def __call__(self, x: float) -> float:
        return parametric_sigmoid(x, self.alpha, self.beta, self.gamma)

    def __repr__(self):
        return f"SigmoidParams(α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f})"


@dataclass
class CubeToSimplexMapping:
    """
    A diffeomorphism from [0,1]^k to the k-simplex using Thurstone racing.

    The mapping works as follows:
    1. Map each x_i ∈ [0,1] to ability a_i using parametric sigmoids
    2. Add a special (k+1)-st horse with fixed ability
    3. Create normal densities for each horse centered at their ability
    4. Run Thurstone race to get winning probabilities → point on k-simplex
    """

    sigmoid_params: List[SigmoidParams]  # One per dimension
    special_horse_ability: float  # Ability of (k+1)-st horse
    noise_scale: float = 1.0  # Standard deviation of performance noise
    lattice: Optional[UniformLattice] = None  # Computational lattice

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
        abilities = np.zeros(self.k + 1)
        for i, (x, sigmoid_param) in enumerate(zip(cube_point, self.sigmoid_params)):
            if not (0 <= x <= 1):
                raise ValueError(f"Cube point coordinate {i} = {x} not in [0,1]")
            abilities[i] = sigmoid_param(x)

        # Set ability of special horse
        abilities[self.k] = self.special_horse_ability

        return abilities

    def create_race(self, cube_point: np.ndarray) -> Race:
        """Create a Race object for the given cube point."""
        abilities = self.cube_to_abilities(cube_point)

        # Create normal densities for each horse
        densities = []
        for ability in abilities:
            # Create normal distribution centered at the horse's ability
            density = Density.skew_normal(
                lattice=self.lattice,
                loc=ability,
                scale=self.noise_scale,
                a=0.0,  # a=0 gives standard normal (not skewed)
            )
            densities.append(density)

        return Race(densities)

    def __call__(self, cube_point: np.ndarray) -> np.ndarray:
        """
        Map point from k-cube to k-simplex.

        Args:
            cube_point: Point in [0,1]^k

        Returns:
            Point on k-simplex (winning probabilities)
        """
        race = self.create_race(cube_point)
        return race.state_prices()

    def forward(self, cube_point: np.ndarray) -> np.ndarray:
        """Alias for __call__ to make direction explicit."""
        return self(cube_point)

    def batch_forward(self, cube_points: np.ndarray) -> np.ndarray:
        """
        Apply mapping to multiple points.

        Args:
            cube_points: Array of shape (n_points, k)

        Returns:
            Array of shape (n_points, k+1) on the simplex
        """
        cube_points = np.asarray(cube_points)
        if cube_points.ndim != 2 or cube_points.shape[1] != self.k:
            raise ValueError(f"Expected shape (n_points, {self.k}), got {cube_points.shape}")

        results = []
        for point in cube_points:
            results.append(self.forward(point))

        return np.array(results)


def create_uniform_sigmoid_params(
    k: int,
    alpha_range: Tuple[float, float] = (0.5, 2.0),
    beta_range: Tuple[float, float] = (2.0, 8.0),
    gamma_range: Tuple[float, float] = (0.2, 0.8),
) -> List[SigmoidParams]:
    """
    Create reasonably uniform sigmoid parameters for k dimensions.

    Args:
        k: Number of dimensions
        alpha_range: Range for scale parameters
        beta_range: Range for steepness parameters
        gamma_range: Range for inflection points

    Returns:
        List of SigmoidParams
    """
    np.random.seed(42)  # For reproducible defaults

    params = []
    for i in range(k):
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        gamma = np.random.uniform(*gamma_range)
        params.append(SigmoidParams(alpha, beta, gamma))

    return params


# Example usage and testing
if __name__ == "__main__":
    print("Testing cube-to-simplex mapping with k=2...")

    # Create a simple mapping for the triangle (k=2)
    sigmoid_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.3),
        SigmoidParams(alpha=1.2, beta=5.0, gamma=0.7),
    ]
    special_ability = 0.0

    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params,
        special_horse_ability=special_ability,
        noise_scale=1.0,
    )

    print(f"Mapping: k={mapping.k}")
    print(f"Sigmoid params: {mapping.sigmoid_params}")

    # Test some points
    test_points = np.array(
        [
            [0.0, 0.0],  # Corner
            [1.0, 1.0],  # Opposite corner
            [0.5, 0.5],  # Center
            [0.2, 0.8],  # Asymmetric
            [0.7, 0.3],  # Another asymmetric
        ]
    )

    print("\nTesting individual points:")
    for i, point in enumerate(test_points):
        simplex_point = mapping(point)
        print(f"Point {i}: {point} → {simplex_point} (sum: {np.sum(simplex_point):.6f})")

    # Test batch processing
    print("\nTesting batch processing:")
    batch_results = mapping.batch_forward(test_points)
    print(f"Batch shape: {batch_results.shape}")
    print(f"All sums close to 1: {np.allclose(np.sum(batch_results, axis=1), 1.0)}")

    # Test abilities computation
    print("\nTesting abilities computation:")
    abilities = mapping.cube_to_abilities([0.3, 0.7])
    print(f"Cube point [0.3, 0.7] → abilities {abilities}")
