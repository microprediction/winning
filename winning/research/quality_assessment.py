"""
Quality assessment tools for cube-to-simplex diffeomorphisms.

This module provides various measures to evaluate the quality of
diffeomorphisms created using the Thurstone racing framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cube_to_simplex import CubeToSimplexMapping


@dataclass
class QualityMetrics:
    """Container for various quality metrics."""

    symmetry_score: float
    volume_preservation_score: Optional[float] = None
    smoothness_score: Optional[float] = None
    coverage_score: Optional[float] = None
    invertibility_score: Optional[float] = None

    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted average of available metrics."""
        if weights is None:
            weights = {
                "symmetry": 1.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 1.0,
                "invertibility": 1.0,
            }

        total_weight = 0.0
        total_score = 0.0

        for metric_name, score in [
            ("symmetry", self.symmetry_score),
            ("volume_preservation", self.volume_preservation_score),
            ("smoothness", self.smoothness_score),
            ("coverage", self.coverage_score),
            ("invertibility", self.invertibility_score),
        ]:
            if score is not None and metric_name in weights:
                weight = weights[metric_name]
                total_score += weight * score
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


def assess_symmetry(
    mapping: CubeToSimplexMapping,
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Assess how symmetric the mapping is - i.e., how close each horse's
    winning probability is to 1/(k+1) when sampling uniformly from the cube.

    Args:
        mapping: The cube-to-simplex mapping to assess
        n_samples: Number of random samples from the cube
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (symmetry_score, details_dict)
        - symmetry_score: Float in [0, 1] where 1 is perfectly symmetric
        - details_dict: Additional information about the assessment
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    k = mapping.k
    expected_prob = 1.0 / (k + 1)

    # Generate random samples from the k-cube
    cube_samples = np.random.uniform(0, 1, size=(n_samples, k))

    # Map to simplex
    simplex_samples = mapping.batch_forward(cube_samples)

    # Compute mean winning probability for each horse
    mean_probs = np.mean(simplex_samples, axis=0)

    # Compute deviations from expected uniform probability
    deviations = np.abs(mean_probs - expected_prob)

    # Symmetry score: 1 - (average relative deviation)
    # Perfect symmetry (all deviations = 0) gives score = 1
    # Maximum possible deviation is (k/(k+1)) - (1/(k+1)) = (k-1)/(k+1)
    max_possible_deviation = (k - 1) / (k + 1)
    avg_relative_deviation = np.mean(deviations) / max_possible_deviation
    symmetry_score = max(0.0, 1.0 - avg_relative_deviation)

    details = {
        "mean_probabilities": mean_probs,
        "expected_probability": expected_prob,
        "deviations": deviations,
        "max_deviation": np.max(deviations),
        "avg_deviation": np.mean(deviations),
        "std_deviation": np.std(deviations),
        "n_samples": n_samples,
    }

    return symmetry_score, details


def assess_volume_preservation(
    mapping: CubeToSimplexMapping,
    n_samples: int = 1000,
    epsilon: float = 1e-4,
    random_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Assess how well the mapping preserves volume by examining the Jacobian determinant.

    For a volume-preserving mapping, |J| should be approximately constant.

    Args:
        mapping: The mapping to assess
        n_samples: Number of random points to sample
        epsilon: Step size for numerical differentiation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (volume_score, details_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    k = mapping.k

    # Generate random samples from cube interior (avoid boundaries)
    margin = 2 * epsilon
    cube_samples = np.random.uniform(margin, 1 - margin, size=(n_samples, k))

    jacobian_dets = []

    for point in cube_samples:
        # Compute Jacobian matrix using finite differences
        jacobian = np.zeros((k + 1, k))  # (k+1) outputs, k inputs

        base_output = mapping(point)

        for j in range(k):
            # Perturb j-th input coordinate
            point_plus = point.copy()
            point_plus[j] += epsilon

            point_minus = point.copy()
            point_minus[j] -= epsilon

            # Central difference approximation
            output_plus = mapping(point_plus)
            output_minus = mapping(point_minus)
            jacobian[:, j] = (output_plus - output_minus) / (2 * epsilon)

        # For mapping to simplex, we need to handle the constraint that outputs sum to 1
        # The effective Jacobian is k x k (remove one redundant dimension)
        effective_jacobian = jacobian[:k, :]  # Take first k rows

        try:
            det = np.linalg.det(effective_jacobian)
            jacobian_dets.append(abs(det))
        except np.linalg.LinAlgError:
            # Skip singular matrices
            continue

    jacobian_dets = np.array(jacobian_dets)

    if len(jacobian_dets) == 0:
        return 0.0, {"error": "No valid Jacobian computations"}

    # Volume preservation score: 1 - coefficient of variation of |det(J)|
    # Perfect preservation would have constant |det(J)|, so CV = 0 → score = 1
    mean_det = np.mean(jacobian_dets)
    std_det = np.std(jacobian_dets)
    cv = std_det / mean_det if mean_det > 0 else float("inf")

    # Score decreases as coefficient of variation increases
    volume_score = max(0.0, 1.0 - min(cv, 1.0))

    details = {
        "jacobian_determinants": jacobian_dets,
        "mean_det": mean_det,
        "std_det": std_det,
        "coefficient_of_variation": cv,
        "min_det": np.min(jacobian_dets),
        "max_det": np.max(jacobian_dets),
        "n_samples": len(jacobian_dets),
    }

    return volume_score, details


def assess_smoothness(
    mapping: CubeToSimplexMapping,
    n_samples: int = 1000,
    epsilon: float = 1e-4,
    random_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Assess smoothness by examining the magnitude of gradients.

    Smoother mappings have more moderate gradients.

    Args:
        mapping: The mapping to assess
        n_samples: Number of points to sample
        epsilon: Step size for numerical differentiation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (smoothness_score, details_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    k = mapping.k

    # Sample from cube interior
    margin = 2 * epsilon
    cube_samples = np.random.uniform(margin, 1 - margin, size=(n_samples, k))

    gradient_norms = []

    for point in cube_samples:
        gradients = []

        base_output = mapping(point)

        # Compute gradient for each output component
        for i in range(k + 1):
            gradient = np.zeros(k)

            for j in range(k):
                # Central difference for partial derivative
                point_plus = point.copy()
                point_plus[j] += epsilon

                point_minus = point.copy()
                point_minus[j] -= epsilon

                output_plus = mapping(point_plus)
                output_minus = mapping(point_minus)

                gradient[j] = (output_plus[i] - output_minus[i]) / (2 * epsilon)

            gradients.append(gradient)

        # Compute norm of gradient vector for each output
        norms = [np.linalg.norm(grad) for grad in gradients]
        gradient_norms.extend(norms)

    gradient_norms = np.array(gradient_norms)

    # Smoothness score: inverse relationship with average gradient magnitude
    # Use a reference scale based on the sigmoid parameters
    mean_alpha = np.mean([param.alpha for param in mapping.sigmoid_params])
    mean_beta = np.mean([param.beta for param in mapping.sigmoid_params])
    reference_scale = mean_alpha * mean_beta  # Rough expected gradient scale

    normalized_gradients = gradient_norms / reference_scale
    mean_normalized_gradient = np.mean(normalized_gradients)

    # Score decreases as gradients get larger
    smoothness_score = max(0.0, 1.0 / (1.0 + mean_normalized_gradient))

    details = {
        "gradient_norms": gradient_norms,
        "mean_gradient_norm": np.mean(gradient_norms),
        "std_gradient_norm": np.std(gradient_norms),
        "max_gradient_norm": np.max(gradient_norms),
        "reference_scale": reference_scale,
        "normalized_mean": mean_normalized_gradient,
        "n_samples": n_samples,
    }

    return smoothness_score, details


def assess_uniform_coverage(
    mapping: CubeToSimplexMapping,
    n_samples: int = 10000,
    n_bins: int = 20,
    random_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Assess how uniformly the mapping covers the simplex.

    Uses a discretization of the simplex to measure coverage uniformity.

    Args:
        mapping: The mapping to assess
        n_samples: Number of uniform samples from the cube
        n_bins: Number of bins per dimension for discretization
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (coverage_score, details_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    k = mapping.k

    # Generate uniform samples from the cube
    cube_samples = np.random.uniform(0, 1, size=(n_samples, k))
    simplex_samples = mapping.batch_forward(cube_samples)

    # For k=2 (triangle), we can create a 2D histogram
    # For higher k, we use a different strategy

    if k == 2:
        # Map triangle to 2D coordinates for binning
        # Use barycentric coordinates (first two components)
        x_coords = simplex_samples[:, 0]
        y_coords = simplex_samples[:, 1]

        # Create 2D histogram
        # Since we're on simplex, we need triangular binning
        # Simple approach: use rectangular binning but mask invalid regions
        hist, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords, bins=n_bins, range=[[0, 1], [0, 1]]
        )

        # Mask bins where x + y > 1 (outside simplex)
        bin_centers_x = (x_edges[:-1] + x_edges[1:]) / 2
        bin_centers_y = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(bin_centers_x, bin_centers_y)
        valid_mask = (X + Y) <= 1.0

        valid_hist = hist[valid_mask]
        n_valid_bins = np.sum(valid_mask)

    else:
        # For higher dimensions, use a simpler approach
        # Divide each coordinate into bins and create a flattened histogram
        hist_coords = []
        for dim in range(k + 1):
            coords = simplex_samples[:, dim]
            hist_coords.append(np.digitize(coords, np.linspace(0, 1, n_bins + 1)) - 1)

        hist_coords = np.array(hist_coords).T  # Shape: (n_samples, k+1)

        # Create flattened histogram
        # This is a simplification - proper simplex binning is complex in high dimensions
        flat_indices = []
        for i, coords in enumerate(hist_coords):
            # Simple hash-based approach
            flat_idx = hash(tuple(coords)) % (n_bins**k)
            flat_indices.append(flat_idx)

        valid_hist, _ = np.histogram(flat_indices, bins=n_bins**k)
        n_valid_bins = n_bins**k

    # Compute coverage uniformity
    if len(valid_hist) == 0 or n_valid_bins == 0:
        return 0.0, {"error": "No valid bins"}

    # Expected count per bin for uniform distribution
    expected_count = n_samples / n_valid_bins

    # Chi-squared like measure of uniformity
    observed_counts = valid_hist[valid_hist > 0]  # Only non-empty bins
    if len(observed_counts) == 0:
        return 0.0, {"error": "No occupied bins"}

    # Coverage score based on how many bins are occupied and how uniform the counts are
    occupancy_ratio = len(observed_counts) / n_valid_bins

    # Uniformity of occupied bins (coefficient of variation)
    mean_count = np.mean(observed_counts)
    std_count = np.std(observed_counts)
    cv_count = std_count / mean_count if mean_count > 0 else float("inf")

    # Combined score: occupancy * uniformity
    occupancy_score = occupancy_ratio
    uniformity_score = max(0.0, 1.0 - min(cv_count / 2.0, 1.0))  # Normalize CV
    coverage_score = (occupancy_score + uniformity_score) / 2.0

    details = {
        "histogram": valid_hist,
        "n_valid_bins": n_valid_bins,
        "n_occupied_bins": len(observed_counts),
        "occupancy_ratio": occupancy_ratio,
        "mean_count": mean_count,
        "std_count": std_count,
        "cv_count": cv_count,
        "occupancy_score": occupancy_score,
        "uniformity_score": uniformity_score,
        "n_samples": n_samples,
    }

    return coverage_score, details


def assess_invertibility(
    mapping: CubeToSimplexMapping,
    n_samples: int = 1000,
    tolerance: float = 0.01,
    max_iterations: int = 100,
    random_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Assess how well-conditioned the inverse mapping is.

    Tests numerical inversion by attempting to recover cube points from simplex points.

    Args:
        mapping: The mapping to assess
        n_samples: Number of test points
        tolerance: Tolerance for successful inversion
        max_iterations: Maximum iterations for numerical inversion
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (invertibility_score, details_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    k = mapping.k

    # Generate test points from the cube
    cube_test_points = np.random.uniform(0, 1, size=(n_samples, k))

    # Map to simplex
    simplex_points = mapping.batch_forward(cube_test_points)

    # Attempt to invert each point using numerical optimization
    successful_inversions = 0
    inversion_errors = []
    condition_numbers = []

    for i, (original_cube, target_simplex) in enumerate(zip(cube_test_points, simplex_points)):
        # Simple gradient descent to find inverse
        # Start from a random point in the cube
        current_cube = np.random.uniform(0, 1, k)

        for iteration in range(max_iterations):
            # Compute current simplex point
            current_simplex = mapping(current_cube)

            # Compute error
            error = np.linalg.norm(current_simplex - target_simplex)

            if error < tolerance:
                successful_inversions += 1
                inversion_error = np.linalg.norm(current_cube - original_cube)
                inversion_errors.append(inversion_error)

                # Estimate condition number using finite differences
                try:
                    epsilon = 1e-6
                    jacobian = np.zeros((k + 1, k))
                    base_output = mapping(current_cube)

                    for j in range(k):
                        if current_cube[j] + epsilon <= 1:
                            perturbed_cube = current_cube.copy()
                            perturbed_cube[j] += epsilon
                            perturbed_output = mapping(perturbed_cube)
                            jacobian[:, j] = (perturbed_output - base_output) / epsilon
                        elif current_cube[j] - epsilon >= 0:
                            perturbed_cube = current_cube.copy()
                            perturbed_cube[j] -= epsilon
                            perturbed_output = mapping(perturbed_cube)
                            jacobian[:, j] = (base_output - perturbed_output) / epsilon

                    # Use the first k rows (since outputs sum to 1)
                    effective_jacobian = jacobian[:k, :]
                    if np.linalg.matrix_rank(effective_jacobian) == k:
                        singular_values = np.linalg.svd(effective_jacobian, compute_uv=False)
                        condition_number = np.max(singular_values) / np.min(singular_values)
                        condition_numbers.append(condition_number)

                except (np.linalg.LinAlgError, ZeroDivisionError):
                    pass
                break

            # Simple gradient step (very basic numerical inversion)
            # In practice, you'd use a more sophisticated optimizer
            if iteration < max_iterations - 1:
                # Compute approximate gradient using finite differences
                gradient = np.zeros(k)
                for j in range(k):
                    if current_cube[j] + 1e-6 <= 1:
                        cube_plus = current_cube.copy()
                        cube_plus[j] += 1e-6
                        simplex_plus = mapping(cube_plus)
                        error_plus = np.linalg.norm(simplex_plus - target_simplex)
                        gradient[j] = (error_plus - error) / 1e-6

                # Gradient descent step
                step_size = 0.01
                current_cube = np.clip(current_cube - step_size * gradient, 0, 1)

    # Compute scores
    success_rate = successful_inversions / n_samples

    if inversion_errors:
        mean_inversion_error = np.mean(inversion_errors)
        # Error score: lower error is better
        error_score = max(0.0, 1.0 - mean_inversion_error)
    else:
        error_score = 0.0
        mean_inversion_error = float("inf")

    if condition_numbers:
        mean_condition_number = np.mean(condition_numbers)
        # Condition score: lower condition number is better
        # Use log scale since condition numbers can vary widely
        log_condition = np.log10(max(mean_condition_number, 1.0))
        condition_score = max(0.0, 1.0 - log_condition / 5.0)  # Penalize log_cond > 5
    else:
        condition_score = 0.0
        mean_condition_number = float("inf")

    # Overall invertibility score
    invertibility_score = (success_rate + error_score + condition_score) / 3.0

    details = {
        "success_rate": success_rate,
        "successful_inversions": successful_inversions,
        "total_attempts": n_samples,
        "mean_inversion_error": mean_inversion_error,
        "inversion_errors": inversion_errors,
        "mean_condition_number": mean_condition_number,
        "condition_numbers": condition_numbers,
        "error_score": error_score,
        "condition_score": condition_score,
    }

    return invertibility_score, details


def comprehensive_quality_assessment(
    mapping: CubeToSimplexMapping,
    symmetry_samples: int = 10000,
    volume_samples: int = 1000,
    smoothness_samples: int = 1000,
    coverage_samples: int = 5000,
    invertibility_samples: int = 100,
    random_seed: Optional[int] = None,
) -> QualityMetrics:
    """
    Perform a comprehensive quality assessment of a cube-to-simplex mapping.

    Args:
        mapping: The mapping to assess
        symmetry_samples: Number of samples for symmetry assessment
        volume_samples: Number of samples for volume preservation
        smoothness_samples: Number of samples for smoothness assessment
        coverage_samples: Number of samples for coverage assessment
        invertibility_samples: Number of samples for invertibility assessment
        random_seed: Random seed for reproducibility

    Returns:
        QualityMetrics object with all computed scores
    """
    print("Assessing symmetry...")
    symmetry_score, _ = assess_symmetry(mapping, symmetry_samples, random_seed)

    print("Assessing volume preservation...")
    volume_score, _ = assess_volume_preservation(mapping, volume_samples, random_seed=random_seed)

    print("Assessing smoothness...")
    smoothness_score, _ = assess_smoothness(mapping, smoothness_samples, random_seed=random_seed)

    print("Assessing uniform coverage...")
    coverage_score, _ = assess_uniform_coverage(mapping, coverage_samples, random_seed=random_seed)

    print("Assessing invertibility...")
    invertibility_score, _ = assess_invertibility(
        mapping, invertibility_samples, random_seed=random_seed
    )

    return QualityMetrics(
        symmetry_score=symmetry_score,
        volume_preservation_score=volume_score,
        smoothness_score=smoothness_score,
        coverage_score=coverage_score,
        invertibility_score=invertibility_score,
    )


# Example usage
if __name__ == "__main__":
    from .cube_to_simplex import CubeToSimplexMapping, SigmoidParams

    print("Testing quality assessment with k=2...")

    # Create a test mapping
    sigmoid_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.3),
        SigmoidParams(alpha=1.2, beta=5.0, gamma=0.7),
    ]
    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params, special_horse_ability=0.0, noise_scale=1.0
    )

    # Test symmetry assessment
    print("\nTesting symmetry assessment:")
    symmetry_score, details = assess_symmetry(mapping, n_samples=5000, random_seed=42)
    print(f"Symmetry score: {symmetry_score:.4f}")
    print(f"Mean probabilities: {details['mean_probabilities']}")
    print(f"Expected: {details['expected_probability']:.4f}")
    print(f"Max deviation: {details['max_deviation']:.4f}")

    # Test comprehensive assessment
    print("\nRunning comprehensive assessment:")
    metrics = comprehensive_quality_assessment(
        mapping,
        symmetry_samples=3000,
        volume_samples=150,
        smoothness_samples=150,
        coverage_samples=2000,
        invertibility_samples=50,
        random_seed=42,
    )
    print(f"Symmetry: {metrics.symmetry_score:.4f}")
    print(f"Volume preservation: {metrics.volume_preservation_score:.4f}")
    print(f"Smoothness: {metrics.smoothness_score:.4f}")
    print(f"Coverage: {metrics.coverage_score:.4f}")
    print(f"Invertibility: {metrics.invertibility_score:.4f}")
    print(f"Overall score: {metrics.overall_score():.4f}")
