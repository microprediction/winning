"""
Hyperparameter optimization for cube-to-simplex diffeomorphisms.

This module provides optimization strategies to find good sigmoid parameters
and special horse abilities that maximize mapping quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .cube_to_simplex import CubeToSimplexMapping, SigmoidParams
from .quality_assessment import QualityMetrics, comprehensive_quality_assessment


@dataclass
class ParameterBounds:
    """Bounds for optimization parameters."""

    alpha_min: float = 0.1
    alpha_max: float = 3.0
    beta_min: float = 1.0
    beta_max: float = 10.0
    gamma_min: float = 0.1
    gamma_max: float = 0.9
    special_ability_min: float = -2.0
    special_ability_max: float = 2.0


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_score: float
    best_mapping: CubeToSimplexMapping
    best_metrics: QualityMetrics
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    optimization_time: float = 0.0


class ObjectiveFunction:
    """
    Objective function for optimization that evaluates mapping quality.
    """

    def __init__(
        self,
        k: int,
        quality_weights: Optional[Dict[str, float]] = None,
        assessment_samples: Dict[str, int] = None,
        noise_scale: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            k: Dimension of the cube/simplex
            quality_weights: Weights for different quality measures
            assessment_samples: Number of samples for each quality assessment
            noise_scale: Standard deviation of performance noise
            random_seed: Random seed for quality assessment
        """
        self.k = k
        self.noise_scale = noise_scale
        self.random_seed = random_seed

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
                "symmetry_samples": 5000,
                "volume_samples": 300,
                "smoothness_samples": 300,
                "coverage_samples": 3000,
                "invertibility_samples": 50,
            }
        self.assessment_samples = assessment_samples

        self.evaluation_count = 0

    def __call__(self, params_vector: np.ndarray) -> float:
        """
        Evaluate objective function for given parameter vector.

        Args:
            params_vector: Flattened parameter vector
                          [alpha_1, beta_1, gamma_1, ..., alpha_k, beta_k, gamma_k, special_ability]

        Returns:
            Objective value (higher is better)
        """
        try:
            self.evaluation_count += 1

            # Parse parameter vector
            sigmoid_params = []
            for i in range(self.k):
                alpha = params_vector[i * 3]
                beta = params_vector[i * 3 + 1]
                gamma = params_vector[i * 3 + 2]
                sigmoid_params.append(SigmoidParams(alpha, beta, gamma))

            special_ability = params_vector[self.k * 3]

            # Create mapping
            mapping = CubeToSimplexMapping(
                sigmoid_params=sigmoid_params,
                special_horse_ability=special_ability,
                noise_scale=self.noise_scale,
            )

            # Assess quality
            metrics = comprehensive_quality_assessment(
                mapping, random_seed=self.random_seed, **self.assessment_samples
            )

            # Compute weighted score
            score = metrics.overall_score(self.quality_weights)

            print(
                f"Evaluation {self.evaluation_count}: Score = {score:.4f} "
                f"(Sym: {metrics.symmetry_score:.3f}, "
                f"Vol: {metrics.volume_preservation_score:.3f}, "
                f"Smooth: {metrics.smoothness_score:.3f}, "
                f"Cov: {metrics.coverage_score:.3f}, "
                f"Inv: {metrics.invertibility_score:.3f})"
            )

            return score

        except Exception as e:
            print(f"Error in evaluation {self.evaluation_count}: {e}")
            return 0.0  # Return poor score on error


class Optimizer(ABC):
    """Base class for optimization algorithms."""

    @abstractmethod
    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        max_evaluations: int,
    ) -> OptimizationResult:
        """Run optimization and return result."""
        pass


class RandomSearchOptimizer(Optimizer):
    """Simple random search optimizer."""

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        max_evaluations: int,
    ) -> OptimizationResult:
        """Run random search optimization."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        k = objective.k
        param_dimension = k * 3 + 1  # 3 params per sigmoid + special ability

        best_score = -np.inf
        best_params_vector = None
        best_mapping = None
        best_metrics = None
        history = []

        print(f"Starting random search optimization with {max_evaluations} evaluations...")

        for i in range(max_evaluations):
            # Generate random parameters within bounds
            params_vector = np.zeros(param_dimension)

            for j in range(k):
                params_vector[j * 3] = np.random.uniform(
                    bounds.alpha_min, bounds.alpha_max
                )  # alpha
                params_vector[j * 3 + 1] = np.random.uniform(
                    bounds.beta_min, bounds.beta_max
                )  # beta
                params_vector[j * 3 + 2] = np.random.uniform(
                    bounds.gamma_min, bounds.gamma_max
                )  # gamma

            params_vector[k * 3] = np.random.uniform(
                bounds.special_ability_min, bounds.special_ability_max
            )

            # Evaluate
            score = objective(params_vector)

            # Update best if improved
            if score > best_score:
                best_score = score
                best_params_vector = params_vector.copy()

                # Create best mapping for result
                sigmoid_params = []
                for j in range(k):
                    alpha = params_vector[j * 3]
                    beta = params_vector[j * 3 + 1]
                    gamma = params_vector[j * 3 + 2]
                    sigmoid_params.append(SigmoidParams(alpha, beta, gamma))

                best_mapping = CubeToSimplexMapping(
                    sigmoid_params=sigmoid_params,
                    special_horse_ability=params_vector[k * 3],
                    noise_scale=objective.noise_scale,
                )

                best_metrics = comprehensive_quality_assessment(
                    best_mapping,
                    random_seed=objective.random_seed,
                    **objective.assessment_samples,
                )

                print(f"*** New best at evaluation {i + 1}: {best_score:.4f} ***")

            # Record history
            history.append(
                {
                    "evaluation": i + 1,
                    "params_vector": params_vector.copy(),
                    "score": score,
                    "is_best": score == best_score,
                }
            )

        # Prepare result
        best_params = {}
        for j in range(k):
            best_params[f"alpha_{j}"] = best_params_vector[j * 3]
            best_params[f"beta_{j}"] = best_params_vector[j * 3 + 1]
            best_params[f"gamma_{j}"] = best_params_vector[j * 3 + 2]
        best_params["special_ability"] = best_params_vector[k * 3]

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_mapping=best_mapping,
            best_metrics=best_metrics,
            optimization_history=history,
            total_evaluations=max_evaluations,
        )


class EvolutionaryOptimizer(Optimizer):
    """Evolutionary algorithm optimizer."""

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        random_seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_seed = random_seed

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        max_evaluations: int,
    ) -> OptimizationResult:
        """Run evolutionary optimization."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        k = objective.k
        param_dimension = k * 3 + 1

        # Initialize population
        population = []
        fitness = []

        print(f"Initializing population of {self.population_size}...")

        for i in range(self.population_size):
            individual = np.zeros(param_dimension)

            for j in range(k):
                individual[j * 3] = np.random.uniform(bounds.alpha_min, bounds.alpha_max)
                individual[j * 3 + 1] = np.random.uniform(bounds.beta_min, bounds.beta_max)
                individual[j * 3 + 2] = np.random.uniform(bounds.gamma_min, bounds.gamma_max)

            individual[k * 3] = np.random.uniform(
                bounds.special_ability_min, bounds.special_ability_max
            )

            population.append(individual)
            fitness.append(objective(individual))

        evaluations_used = self.population_size
        generation = 0
        history = []
        best_score = max(fitness)
        best_individual = population[np.argmax(fitness)].copy()

        print(f"Initial best fitness: {best_score:.4f}")

        # Evolution loop
        while evaluations_used < max_evaluations:
            generation += 1
            print(f"Generation {generation} (evaluations: {evaluations_used}/{max_evaluations})")

            new_population = []
            new_fitness = []

            # Elitism: keep best individual
            best_idx = np.argmax(fitness)
            new_population.append(population[best_idx].copy())
            new_fitness.append(fitness[best_idx])

            # Generate rest of population
            while len(new_population) < self.population_size and evaluations_used < max_evaluations:
                if np.random.random() < self.crossover_rate and len(population) >= 2:
                    # Crossover
                    parent1_idx = self._tournament_selection(fitness)
                    parent2_idx = self._tournament_selection(fitness)
                    child = self._crossover(population[parent1_idx], population[parent2_idx])
                else:
                    # Mutation of random individual
                    parent_idx = np.random.randint(len(population))
                    child = population[parent_idx].copy()

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child, bounds)

                # Ensure bounds
                child = self._clip_to_bounds(child, bounds, k)

                # Evaluate child
                child_fitness = objective(child)
                evaluations_used += 1

                new_population.append(child)
                new_fitness.append(child_fitness)

                # Update global best
                if child_fitness > best_score:
                    best_score = child_fitness
                    best_individual = child.copy()
                    print(f"*** New best in generation {generation}: {best_score:.4f} ***")

            population = new_population
            fitness = new_fitness

            # Record history
            history.append(
                {
                    "generation": generation,
                    "best_fitness": best_score,
                    "mean_fitness": np.mean(fitness),
                    "evaluations": evaluations_used,
                }
            )

        # Create result
        best_params = {}
        for j in range(k):
            best_params[f"alpha_{j}"] = best_individual[j * 3]
            best_params[f"beta_{j}"] = best_individual[j * 3 + 1]
            best_params[f"gamma_{j}"] = best_individual[j * 3 + 2]
        best_params["special_ability"] = best_individual[k * 3]

        # Create best mapping
        sigmoid_params = []
        for j in range(k):
            alpha = best_individual[j * 3]
            beta = best_individual[j * 3 + 1]
            gamma = best_individual[j * 3 + 2]
            sigmoid_params.append(SigmoidParams(alpha, beta, gamma))

        best_mapping = CubeToSimplexMapping(
            sigmoid_params=sigmoid_params,
            special_horse_ability=best_individual[k * 3],
            noise_scale=objective.noise_scale,
        )

        best_metrics = comprehensive_quality_assessment(
            best_mapping,
            random_seed=objective.random_seed,
            **objective.assessment_samples,
        )

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_mapping=best_mapping,
            best_metrics=best_metrics,
            optimization_history=history,
            total_evaluations=evaluations_used,
        )

    def _tournament_selection(self, fitness: List[float], tournament_size: int = 3) -> int:
        """Tournament selection for parent selection."""
        candidates = np.random.choice(
            len(fitness), size=min(tournament_size, len(fitness)), replace=False
        )
        best_candidate = candidates[np.argmax([fitness[i] for i in candidates])]
        return best_candidate

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def _mutate(self, individual: np.ndarray, bounds: ParameterBounds) -> np.ndarray:
        """Gaussian mutation."""
        k = (len(individual) - 1) // 3
        mutated = individual.copy()

        # Mutate each parameter with some probability
        for i in range(len(individual)):
            if np.random.random() < 0.3:  # 30% chance to mutate each parameter
                if i % 3 == 0:  # Alpha parameter
                    std = (bounds.alpha_max - bounds.alpha_min) * 0.1
                    mutated[i] += np.random.normal(0, std)
                elif i % 3 == 1:  # Beta parameter
                    std = (bounds.beta_max - bounds.beta_min) * 0.1
                    mutated[i] += np.random.normal(0, std)
                elif i % 3 == 2:  # Gamma parameter
                    std = (bounds.gamma_max - bounds.gamma_min) * 0.1
                    mutated[i] += np.random.normal(0, std)
                else:  # Special ability
                    std = (bounds.special_ability_max - bounds.special_ability_min) * 0.1
                    mutated[i] += np.random.normal(0, std)

        return mutated

    def _clip_to_bounds(
        self, individual: np.ndarray, bounds: ParameterBounds, k: int
    ) -> np.ndarray:
        """Clip parameters to bounds."""
        clipped = individual.copy()

        for j in range(k):
            clipped[j * 3] = np.clip(clipped[j * 3], bounds.alpha_min, bounds.alpha_max)
            clipped[j * 3 + 1] = np.clip(clipped[j * 3 + 1], bounds.beta_min, bounds.beta_max)
            clipped[j * 3 + 2] = np.clip(clipped[j * 3 + 2], bounds.gamma_min, bounds.gamma_max)

        clipped[k * 3] = np.clip(
            clipped[k * 3], bounds.special_ability_min, bounds.special_ability_max
        )

        return clipped


def optimize_diffeomorphism(
    k: int,
    optimizer: str = "random",
    max_evaluations: int = 100,
    bounds: Optional[ParameterBounds] = None,
    quality_weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
) -> OptimizationResult:
    """
    Optimize a k-dimensional cube-to-simplex diffeomorphism.

    Args:
        k: Dimension of cube and simplex
        optimizer: 'random' or 'evolutionary'
        max_evaluations: Maximum number of function evaluations
        bounds: Parameter bounds
        quality_weights: Weights for quality measures
        random_seed: Random seed for reproducibility

    Returns:
        OptimizationResult with best found parameters
    """
    if bounds is None:
        bounds = ParameterBounds()

    # Create objective function
    objective = ObjectiveFunction(k=k, quality_weights=quality_weights, random_seed=random_seed)

    # Create optimizer
    if optimizer == "random":
        opt = RandomSearchOptimizer(random_seed=random_seed)
    elif optimizer == "evolutionary":
        opt = EvolutionaryOptimizer(random_seed=random_seed)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Run optimization
    print(f"Optimizing {k}-dimensional diffeomorphism using {optimizer} search...")
    result = opt.optimize(objective, bounds, max_evaluations)

    print("\nOptimization complete!")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")

    return result


# Example usage
if __name__ == "__main__":
    print("Testing hyperparameter optimization for k=2...")

    # Quick test with small evaluation budget
    result = optimize_diffeomorphism(
        k=2,
        optimizer="random",
        max_evaluations=20,
        quality_weights={"symmetry": 2.0, "smoothness": 1.0},
        random_seed=42,
    )

    print("\nFinal Results:")
    print(f"Best overall score: {result.best_score:.4f}")
    print("Quality breakdown:")
    print(f"  Symmetry: {result.best_metrics.symmetry_score:.4f}")
    print(f"  Volume preservation: {result.best_metrics.volume_preservation_score:.4f}")
    print(f"  Smoothness: {result.best_metrics.smoothness_score:.4f}")
    print(f"  Coverage: {result.best_metrics.coverage_score:.4f}")
    print(f"  Invertibility: {result.best_metrics.invertibility_score:.4f}")

    print("\nBest sigmoid parameters:")
    for i, param in enumerate(result.best_mapping.sigmoid_params):
        print(f"  Dimension {i}: {param}")
    print(f"Special horse ability: {result.best_mapping.special_horse_ability:.4f}")
