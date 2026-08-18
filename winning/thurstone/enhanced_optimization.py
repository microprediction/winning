"""
Enhanced optimization for Thurstone diffeomorphisms using pure Python optimizers.

This module integrates the pure optimizer collection with our diffeomorphism
framework to find optimal parameter configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .adaptive_special_horse import (
    AdaptiveSpecialHorse,
    DistributionType,
    SpecialHorseConfig,
)
from .cube_to_simplex import SigmoidParams
from .enhanced_cube_to_simplex import EnhancedCubeToSimplexMapping
from .pure_optimizers import pure_optimize
from .quality_assessment import QualityMetrics, comprehensive_quality_assessment


@dataclass
class EnhancedOptimizationResult:
    """Result of enhanced diffeomorphism optimization."""

    best_params: Dict[str, Any]
    best_score: float
    best_mapping: EnhancedCubeToSimplexMapping
    best_metrics: QualityMetrics
    algorithm_used: str
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    parameter_bounds: Dict[str, Tuple[float, float]]


class EnhancedDiffeomorphismObjective:
    """
    Enhanced objective function that optimizes both sigmoid parameters
    and special horse configuration.
    """

    def __init__(
        self,
        k: int,
        quality_weights: Optional[Dict[str, float]] = None,
        assessment_samples: Optional[Dict[str, int]] = None,
        optimize_special_horse: bool = True,
        noise_scale: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            k: Dimension of cube/simplex
            quality_weights: Weights for quality measures
            assessment_samples: Sample sizes for quality assessment
            optimize_special_horse: Whether to optimize special horse params
            noise_scale: Noise scale for performance
            random_seed: Random seed for quality assessment
        """
        self.k = k
        self.noise_scale = noise_scale
        self.random_seed = random_seed
        self.optimize_special_horse = optimize_special_horse

        if quality_weights is None:
            quality_weights = {
                "symmetry": 2.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 1.5,
                "invertibility": 1.0,
            }
        self.quality_weights = quality_weights

        if assessment_samples is None:
            assessment_samples = {
                "symmetry_samples": 2000,
                "volume_samples": 150,
                "smoothness_samples": 150,
                "coverage_samples": 1200,
                "invertibility_samples": 25,
            }
        self.assessment_samples = assessment_samples

        # Define parameter structure
        self.param_structure = self._define_parameter_structure()
        self.param_dimension = len(self.param_structure)
        self.evaluation_count = 0

    def _define_parameter_structure(self) -> List[Tuple[str, str, Tuple[float, float]]]:
        """Define the parameter structure and bounds."""
        structure = []

        # Sigmoid parameters for each dimension
        for i in range(self.k):
            structure.append((f"sigmoid_{i}_alpha", "sigmoid", (0.1, 3.0)))
            structure.append((f"sigmoid_{i}_beta", "sigmoid", (1.0, 10.0)))
            structure.append((f"sigmoid_{i}_gamma", "sigmoid", (0.1, 0.9)))

        if self.optimize_special_horse:
            # Special horse parameters
            structure.append(("special_base_ability", "special", (-2.0, 2.0)))
            structure.append(("special_location", "special", (-1.0, 1.0)))
            structure.append(("special_scale", "special", (0.1, 3.0)))
            structure.append(
                ("special_distribution_type", "special", (0, 4))
            )  # 0-4 for distribution types

        return structure

    def _params_vector_to_config(
        self, x: np.ndarray
    ) -> Tuple[List[SigmoidParams], SpecialHorseConfig]:
        """Convert optimization parameter vector to configuration objects."""
        # Extract sigmoid parameters
        sigmoid_params = []
        for i in range(self.k):
            alpha_idx = i * 3
            beta_idx = i * 3 + 1
            gamma_idx = i * 3 + 2

            # Map from [0,1] to actual parameter ranges
            alpha_bounds = (0.1, 3.0)
            beta_bounds = (1.0, 10.0)
            gamma_bounds = (0.1, 0.9)

            alpha = alpha_bounds[0] + x[alpha_idx] * (alpha_bounds[1] - alpha_bounds[0])
            beta = beta_bounds[0] + x[beta_idx] * (beta_bounds[1] - beta_bounds[0])
            gamma = gamma_bounds[0] + x[gamma_idx] * (gamma_bounds[1] - gamma_bounds[0])

            sigmoid_params.append(SigmoidParams(alpha, beta, gamma))

        # Extract special horse parameters
        if self.optimize_special_horse:
            special_start_idx = self.k * 3

            # Map special horse parameters
            ability_bounds = (-2.0, 2.0)
            location_bounds = (-1.0, 1.0)
            scale_bounds = (0.1, 3.0)

            base_ability = ability_bounds[0] + x[special_start_idx] * (
                ability_bounds[1] - ability_bounds[0]
            )
            location = location_bounds[0] + x[special_start_idx + 1] * (
                location_bounds[1] - location_bounds[0]
            )
            scale = scale_bounds[0] + x[special_start_idx + 2] * (scale_bounds[1] - scale_bounds[0])

            # Distribution type (discrete choice)
            dist_idx = int(x[special_start_idx + 3] * 5)  # 0-4
            dist_types = [
                DistributionType.NORMAL,
                DistributionType.STUDENT_T,
                DistributionType.LAPLACE,
                DistributionType.UNIFORM,
                DistributionType.EXPONENTIAL,
            ]
            distribution = dist_types[min(dist_idx, len(dist_types) - 1)]

            special_horse_config = SpecialHorseConfig(
                distribution=distribution,
                base_ability=base_ability,
                location=location,
                scale=scale,
                shape=3.0 if distribution == DistributionType.STUDENT_T else 1.0,
            )
        else:
            # Default special horse config
            special_horse_config = SpecialHorseConfig()

        return sigmoid_params, special_horse_config

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective function for given parameter vector.

        Args:
            x: Parameter vector in [0,1]^n

        Returns:
            Objective value (lower is better)
        """
        try:
            self.evaluation_count += 1

            # Convert parameter vector to configurations
            sigmoid_params, special_horse_config = self._params_vector_to_config(x)

            # Create enhanced mapping
            special_horse = AdaptiveSpecialHorse(special_horse_config)
            mapping = EnhancedCubeToSimplexMapping(
                sigmoid_params=sigmoid_params,
                special_horse=special_horse,
                noise_scale=self.noise_scale,
            )

            # Assess quality
            metrics = comprehensive_quality_assessment(
                mapping, random_seed=self.random_seed, **self.assessment_samples
            )

            # Compute weighted score
            score = metrics.overall_score(self.quality_weights)

            # Return negative score (since optimizers minimize)
            return -score

        except Exception as e:
            print(f"Error in evaluation {self.evaluation_count}: {e}")
            return 1.0  # Return poor score on error


def enhanced_optimize_diffeomorphism(
    k: int,
    algorithm: str = "HarmonySearch",
    max_evaluations: int = 100,
    quality_weights: Optional[Dict[str, float]] = None,
    optimize_special_horse: bool = True,
    random_seed: Optional[int] = None,
) -> EnhancedOptimizationResult:
    """
    Optimize diffeomorphism using enhanced framework with pure optimizers.

    Args:
        k: Dimension of cube/simplex
        algorithm: Optimizer to use from PURE_OPTIMIZERS
        max_evaluations: Maximum number of evaluations
        quality_weights: Weights for quality measures
        optimize_special_horse: Whether to optimize special horse
        random_seed: Random seed for reproducibility

    Returns:
        EnhancedOptimizationResult with best configuration
    """
    import time

    print("ENHANCED DIFFEOMORPHISM OPTIMIZATION")
    print(f"   Algorithm: {algorithm}")
    print(f"   Dimension: k={k}")
    print(f"   Max evaluations: {max_evaluations}")
    print(f"   Optimize special horse: {optimize_special_horse}")

    # Create objective function
    objective = EnhancedDiffeomorphismObjective(
        k=k,
        quality_weights=quality_weights,
        optimize_special_horse=optimize_special_horse,
        random_seed=random_seed,
    )

    # Set up optimization
    n_dim = objective.param_dimension
    print(f"   Parameter dimension: {n_dim}")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Run optimization
    start_time = time.time()

    def tracked_objective(x):
        result = objective(x)
        if objective.evaluation_count % max(1, max_evaluations // 10) == 0:
            print(f"   Evaluation {objective.evaluation_count}: Score = {-result:.4f}")
        return result

    best_value, best_x = pure_optimize(
        objective=tracked_objective,
        algorithm=algorithm,
        n_trials=max_evaluations,
        n_dim=n_dim,
    )

    optimization_time = time.time() - start_time

    # Create best mapping from results
    sigmoid_params, special_horse_config = objective._params_vector_to_config(best_x)
    special_horse = AdaptiveSpecialHorse(special_horse_config)
    best_mapping = EnhancedCubeToSimplexMapping(
        sigmoid_params=sigmoid_params,
        special_horse=special_horse,
        noise_scale=objective.noise_scale,
    )

    # Final quality assessment
    print("   Computing final quality assessment...")
    best_metrics = comprehensive_quality_assessment(
        best_mapping, random_seed=random_seed, **objective.assessment_samples
    )

    # Prepare parameter dictionary
    best_params = {}
    for i, (name, param_type, bounds) in enumerate(objective.param_structure):
        # Map back from [0,1] to original bounds
        value = bounds[0] + best_x[i] * (bounds[1] - bounds[0])
        if name == "special_distribution_type":
            # Handle discrete distribution type
            dist_types = ["normal", "student_t", "laplace", "uniform", "exponential"]
            dist_idx = int(value)
            best_params[name] = dist_types[min(dist_idx, len(dist_types) - 1)]
        else:
            best_params[name] = value

    # Create result object
    result = EnhancedOptimizationResult(
        best_params=best_params,
        best_score=-best_value,  # Convert back to positive score
        best_mapping=best_mapping,
        best_metrics=best_metrics,
        algorithm_used=algorithm,
        optimization_history=[],  # Could be extended to track history
        total_evaluations=objective.evaluation_count,
        optimization_time=optimization_time,
        parameter_bounds={name: bounds for name, _, bounds in objective.param_structure},
    )

    print("\n✅ OPTIMIZATION COMPLETE!")
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Total evaluations: {result.total_evaluations}")
    print(f"   Runtime: {result.optimization_time:.1f}s")

    return result


def compare_optimization_algorithms(
    k: int = 2,
    algorithms: Optional[List[str]] = None,
    max_evaluations: int = 50,
    quality_weights: Optional[Dict[str, float]] = None,
    n_runs: int = 3,
) -> Dict[str, List[EnhancedOptimizationResult]]:
    """
    Compare different optimization algorithms on the diffeomorphism problem.

    Args:
        k: Dimension of problem
        algorithms: List of algorithms to compare
        max_evaluations: Evaluations per run
        quality_weights: Quality measure weights
        n_runs: Number of runs per algorithm

    Returns:
        Dictionary mapping algorithm names to results
    """
    if algorithms is None:
        algorithms = ["HarmonySearch", "DifferentialEvolution", "ParticleSwarm"]

    print("ALGORITHM COMPARISON")
    print(f"   Algorithms: {', '.join(algorithms)}")
    print(f"   Runs per algorithm: {n_runs}")
    print(f"   Evaluations per run: {max_evaluations}")

    results = {}

    for algorithm in algorithms:
        print(f"\nTesting {algorithm}...")
        results[algorithm] = []

        for run in range(n_runs):
            print(f"   Run {run + 1}/{n_runs}")

            result = enhanced_optimize_diffeomorphism(
                k=k,
                algorithm=algorithm,
                max_evaluations=max_evaluations,
                quality_weights=quality_weights,
                random_seed=42 + run,
            )

            results[algorithm].append(result)
            print(f"   → Score: {result.best_score:.4f}")

        # Summary statistics
        scores = [r.best_score for r in results[algorithm]]
        print(f"   {algorithm} summary: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return results


# Example usage
if __name__ == "__main__":
    print("ENHANCED DIFFEOMORPHISM OPTIMIZATION TEST")
    print("=" * 60)

    # Test single optimization
    result = enhanced_optimize_diffeomorphism(
        k=2,
        algorithm="HarmonySearch",
        max_evaluations=30,
        quality_weights={"symmetry": 2.0, "coverage": 1.5},
        optimize_special_horse=True,
        random_seed=42,
    )

    print("\nOPTIMIZATION RESULTS:")
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Algorithm: {result.algorithm_used}")
    print(f"   Evaluations: {result.total_evaluations}")

    print("\nSpecial horse configuration:")
    special_config = result.best_mapping.special_horse.config
    print(f"   Distribution: {special_config.distribution.value}")
    print(f"   Base ability: {special_config.base_ability:.3f}")
    print(f"   Scale: {special_config.scale:.3f}")

    print("\nQuality breakdown:")
    metrics = result.best_metrics
    print(f"   Symmetry: {metrics.symmetry_score:.4f}")
    print(f"   Volume preservation: {(metrics.volume_preservation_score or 0):.4f}")
    print(f"   Smoothness: {(metrics.smoothness_score or 0):.4f}")
    print(f"   Coverage: {(metrics.coverage_score or 0):.4f}")
    print(f"   Invertibility: {(metrics.invertibility_score or 0):.4f}")

    print("\n✅ Enhanced optimization framework ready!")
