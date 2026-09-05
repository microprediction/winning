"""
Visualization tools for cube-to-simplex diffeomorphisms.

This module provides plotting functions to visualize mappings,
lattice point transformations, and quality measures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional matplotlib import with graceful fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from .cube_to_simplex import CubeToSimplexMapping
from .quality_assessment import (
    assess_smoothness,
    assess_symmetry,
    assess_uniform_coverage,
    assess_volume_preservation,
)


def _ensure_matplotlib():
    """Ensure matplotlib is available for visualization functions."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization functions. "
            "Install with: pip install 'thurstone[viz]' or pip install matplotlib"
        )


def create_lattice_grid(resolution: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a regular lattice grid in [0,1]^2.

    Args:
        resolution: Number of points per dimension

    Returns:
        Tuple of (x_coords, y_coords) meshgrid arrays
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_cube_lattice_2d(
    mapping: CubeToSimplexMapping,
    resolution: int = 20,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot 2D visualization of cube lattice and its simplex image for k=2.

    Args:
        mapping: The mapping to visualize
        resolution: Grid resolution
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    _ensure_matplotlib()

    if mapping.k != 2:
        raise ValueError("This function is only for k=2 (triangle)")

    # Create lattice points
    X, Y = create_lattice_grid(resolution)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])

    # Map to simplex
    simplex_points = mapping.batch_forward(cube_points)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Original lattice in cube
    ax1.scatter(X.ravel(), Y.ravel(), c="blue", alpha=0.6, s=20)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("Lattice Points in Unit Square [0,1]²")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot 2: Mapped points on simplex (projected to 2D)
    # For triangle, we can show coordinates (p₁, p₂) with constraint p₁ + p₂ + p₃ = 1
    p1, p2, p3 = simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2]

    # Color points by their third coordinate (p₃)
    scatter = ax2.scatter(p1, p2, c=p3, cmap="viridis", alpha=0.7, s=20)

    # Draw simplex boundary (triangle)
    triangle_x = [0, 1, 0, 0]
    triangle_y = [0, 0, 1, 0]
    ax2.plot(triangle_x, triangle_y, "k--", linewidth=2, alpha=0.8)

    # Shade the valid simplex region
    triangle_fill_x = [0, 1, 0]
    triangle_fill_y = [0, 0, 1]
    ax2.fill(triangle_fill_x, triangle_fill_y, alpha=0.1, color="gray")

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("p₁ (Horse 1 win probability)")
    ax2.set_ylabel("p₂ (Horse 2 win probability)")
    ax2.set_title("Mapped Points on 2-Simplex")
    ax2.set_aspect("equal")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("p₃ (Horse 3 win probability)")

    plt.tight_layout()
    return fig


def plot_cube_to_simplex_3d(
    mapping: CubeToSimplexMapping,
    resolution: int = 15,
    figsize: Tuple[float, float] = (15, 5),
) -> plt.Figure:
    """
    Plot 3D visualization showing cube points and their simplex images.

    Args:
        mapping: The mapping to visualize
        resolution: Grid resolution
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if mapping.k != 2:
        raise ValueError("This function is only for k=2 (triangle)")

    # Create lattice points
    X, Y = create_lattice_grid(resolution)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])
    simplex_points = mapping.batch_forward(cube_points)

    # Create figure with 3D subplots
    fig = plt.figure(figsize=figsize)

    # Plot 1: Cube points in 3D (with z=0)
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(X.ravel(), Y.ravel(), np.zeros_like(X.ravel()), c="blue", alpha=0.6, s=20)
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_zlabel("0")
    ax1.set_title("Cube Points [0,1]²")

    # Plot 2: Simplex points in 3D (barycentric coordinates)
    ax2 = fig.add_subplot(132, projection="3d")
    p1, p2, p3 = simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2]
    scatter = ax2.scatter(p1, p2, p3, c=np.arange(len(p1)), cmap="plasma", alpha=0.7, s=20)

    # Draw simplex vertices and edges
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax2.scatter(*vertices.T, c="red", s=100, marker="o")

    # Draw simplex edges
    edges = [(0, 1), (0, 2), (1, 2)]
    for edge in edges:
        ax2.plot3D(*vertices[list(edge)].T, "k-", alpha=0.5)

    ax2.set_xlabel("p₁")
    ax2.set_ylabel("p₂")
    ax2.set_zlabel("p₃")
    ax2.set_title("Mapped Points on 2-Simplex")

    # Plot 3: Combined view with connecting lines
    ax3 = fig.add_subplot(133, projection="3d")

    # Show cube points at z=0
    ax3.scatter(
        X.ravel(),
        Y.ravel(),
        np.zeros_like(X.ravel()),
        c="blue",
        alpha=0.4,
        s=15,
        label="Cube points",
    )

    # Show simplex points at z=1 level
    ax3.scatter(p1, p2, p3 + 1, c="red", alpha=0.4, s=15, label="Simplex points")

    # Draw sample connecting lines
    n_connections = min(50, len(cube_points))
    indices = np.random.choice(len(cube_points), n_connections, replace=False)
    for i in indices:
        x_cube, y_cube = cube_points[i]
        x_simp, y_simp, z_simp = simplex_points[i]
        ax3.plot3D(
            [x_cube, x_simp],
            [y_cube, y_simp],
            [0, z_simp + 1],
            "gray",
            alpha=0.3,
            linewidth=0.5,
        )

    ax3.set_xlabel("x / p₁")
    ax3.set_ylabel("y / p₂")
    ax3.set_zlabel("Level")
    ax3.set_title("Cube → Simplex Mapping")
    ax3.legend()

    plt.tight_layout()
    return fig


def plot_jacobian_heatmap(
    mapping: CubeToSimplexMapping,
    resolution: int = 30,
    epsilon: float = 1e-4,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot heatmap of Jacobian determinant across the cube domain.

    Args:
        mapping: The mapping to visualize
        resolution: Grid resolution
        epsilon: Step size for finite differences
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if mapping.k != 2:
        raise ValueError("This function is only for k=2")

    # Create grid
    margin = 2 * epsilon
    x = np.linspace(margin, 1 - margin, resolution)
    y = np.linspace(margin, 1 - margin, resolution)
    X, Y = np.meshgrid(x, y)

    # Compute Jacobian determinant at each point
    jacobian_dets = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])

            # Compute Jacobian using finite differences
            jacobian = np.zeros((3, 2))  # 3 outputs, 2 inputs

            base_output = mapping(point)

            # Partial derivatives w.r.t. x₁
            point_x_plus = point + np.array([epsilon, 0])
            point_x_minus = point + np.array([-epsilon, 0])
            output_x_plus = mapping(point_x_plus)
            output_x_minus = mapping(point_x_minus)
            jacobian[:, 0] = (output_x_plus - output_x_minus) / (2 * epsilon)

            # Partial derivatives w.r.t. x₂
            point_y_plus = point + np.array([0, epsilon])
            point_y_minus = point + np.array([0, -epsilon])
            output_y_plus = mapping(point_y_plus)
            output_y_minus = mapping(point_y_minus)
            jacobian[:, 1] = (output_y_plus - output_y_minus) / (2 * epsilon)

            # Use 2x2 submatrix (since outputs sum to 1)
            effective_jacobian = jacobian[:2, :]
            try:
                det = np.linalg.det(effective_jacobian)
                jacobian_dets[i, j] = abs(det)
            except np.linalg.LinAlgError:
                jacobian_dets[i, j] = 0

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Jacobian determinant heatmap
    im1 = ax1.imshow(
        jacobian_dets,
        extent=[margin, 1 - margin, margin, 1 - margin],
        origin="lower",
        cmap="plasma",
        aspect="equal",
    )
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("|det(J)| - Jacobian Determinant")
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Log-scale version for better visualization
    log_dets = np.log10(jacobian_dets + 1e-10)
    im2 = ax2.imshow(
        log_dets,
        extent=[margin, 1 - margin, margin, 1 - margin],
        origin="lower",
        cmap="plasma",
        aspect="equal",
    )
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_title("log₁₀|det(J)| - Log Scale")
    plt.colorbar(im2, ax=ax2)

    # Add statistics
    mean_det = np.mean(jacobian_dets)
    std_det = np.std(jacobian_dets)
    fig.suptitle(f"Jacobian Analysis (Mean: {mean_det:.3f}, Std: {std_det:.3f})")

    plt.tight_layout()
    return fig


def plot_quality_summary(
    mapping: CubeToSimplexMapping,
    assessment_samples: Optional[Dict[str, int]] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Create a comprehensive quality summary plot.

    Args:
        mapping: The mapping to assess
        assessment_samples: Sample sizes for quality assessment
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if assessment_samples is None:
        assessment_samples = {
            "symmetry_samples": 5000,
            "volume_samples": 500,
            "smoothness_samples": 500,
            "coverage_samples": 3000,
            "invertibility_samples": 100,
        }

    print("Computing quality metrics for visualization...")

    # Compute quality metrics
    symmetry_score, symmetry_details = assess_symmetry(
        mapping, assessment_samples["symmetry_samples"], random_seed=42
    )

    volume_score, volume_details = assess_volume_preservation(
        mapping, assessment_samples["volume_samples"], random_seed=42
    )

    smoothness_score, smoothness_details = assess_smoothness(
        mapping, assessment_samples["smoothness_samples"], random_seed=42
    )

    coverage_score, coverage_details = assess_uniform_coverage(
        mapping, assessment_samples["coverage_samples"], random_seed=42
    )

    # Create summary plot
    fig = plt.figure(figsize=figsize)

    # Plot 1: Quality scores bar chart
    ax1 = plt.subplot(2, 3, 1)
    scores = [symmetry_score, volume_score, smoothness_score, coverage_score]
    labels = ["Symmetry", "Volume\nPreserv.", "Smoothness", "Coverage"]
    colors = ["blue", "green", "orange", "red"]

    bars = ax1.bar(labels, scores, color=colors, alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Quality Score")
    ax1.set_title("Quality Metrics Summary")
    ax1.grid(True, alpha=0.3)

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 2: Symmetry details
    ax2 = plt.subplot(2, 3, 2)
    mean_probs = symmetry_details["mean_probabilities"]
    expected_prob = symmetry_details["expected_probability"]

    horse_labels = [f"Horse {i + 1}" for i in range(len(mean_probs))]
    ax2.bar(horse_labels, mean_probs, alpha=0.7, color="skyblue")
    ax2.axhline(
        y=expected_prob,
        color="red",
        linestyle="--",
        label=f"Expected: {expected_prob:.3f}",
    )
    ax2.set_ylabel("Win Probability")
    ax2.set_title("Symmetry: Win Probabilities")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Volume preservation histogram
    ax3 = plt.subplot(2, 3, 3)
    if "jacobian_determinants" in volume_details:
        jacobian_dets = volume_details["jacobian_determinants"]
        ax3.hist(jacobian_dets, bins=30, alpha=0.7, color="green")
        ax3.axvline(
            x=np.mean(jacobian_dets),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(jacobian_dets):.3f}",
        )
        ax3.set_xlabel("|det(J)|")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Volume: Jacobian Determinants")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Smoothness gradient distribution
    ax4 = plt.subplot(2, 3, 4)
    if "gradient_norms" in smoothness_details:
        gradient_norms = smoothness_details["gradient_norms"]
        ax4.hist(gradient_norms, bins=30, alpha=0.7, color="orange")
        ax4.axvline(
            x=np.mean(gradient_norms),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(gradient_norms):.3f}",
        )
        ax4.set_xlabel("Gradient Norm")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Smoothness: Gradient Magnitudes")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Coverage visualization (for k=2)
    ax5 = plt.subplot(2, 3, 5)
    if mapping.k == 2:
        # Sample points and show their distribution
        cube_samples = np.random.uniform(0, 1, (1000, 2))
        simplex_samples = mapping.batch_forward(cube_samples)

        ax5.scatter(simplex_samples[:, 0], simplex_samples[:, 1], alpha=0.5, s=10, c="purple")

        # Draw simplex boundary
        triangle_x = [0, 1, 0, 0]
        triangle_y = [0, 0, 1, 0]
        ax5.plot(triangle_x, triangle_y, "k--", linewidth=2)
        ax5.fill([0, 1, 0], [0, 0, 1], alpha=0.1, color="gray")

        ax5.set_xlim(-0.05, 1.05)
        ax5.set_ylim(-0.05, 1.05)
        ax5.set_xlabel("p₁")
        ax5.set_ylabel("p₂")
        ax5.set_title("Coverage: Point Distribution")
        ax5.set_aspect("equal")

    # Plot 6: Parameter visualization
    ax6 = plt.subplot(2, 3, 6)
    param_names = []
    param_values = []

    for i, param in enumerate(mapping.sigmoid_params):
        param_names.extend([f"α{i + 1}", f"β{i + 1}", f"γ{i + 1}"])
        param_values.extend([param.alpha, param.beta, param.gamma])

    param_names.append("Special\nAbility")
    param_values.append(mapping.special_horse_ability)

    ax6.bar(range(len(param_names)), param_values, alpha=0.7, color="teal")
    ax6.set_xticks(range(len(param_names)))
    ax6.set_xticklabels(param_names, rotation=45, ha="right")
    ax6.set_ylabel("Parameter Value")
    ax6.set_title("Model Parameters")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_mapping_comprehensive(
    mapping: CubeToSimplexMapping, resolution: int = 20, save_path: Optional[str] = None
) -> List[plt.Figure]:
    """
    Create comprehensive visualization of a cube-to-simplex mapping.

    Args:
        mapping: The mapping to visualize
        resolution: Grid resolution for plots
        save_path: Optional path to save figures

    Returns:
        List of created figures
    """
    figures = []

    print("Creating lattice visualization...")
    fig1 = plot_cube_lattice_2d(mapping, resolution)
    figures.append(fig1)

    print("Creating 3D visualization...")
    fig2 = plot_cube_to_simplex_3d(mapping, resolution)
    figures.append(fig2)

    print("Creating Jacobian analysis...")
    fig3 = plot_jacobian_heatmap(mapping, resolution // 2)  # Lower resolution for speed
    figures.append(fig3)

    print("Creating quality summary...")
    fig4 = plot_quality_summary(mapping)
    figures.append(fig4)

    if save_path:
        fig1.savefig(f"{save_path}_lattice.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"{save_path}_3d.png", dpi=300, bbox_inches="tight")
        fig3.savefig(f"{save_path}_jacobian.png", dpi=300, bbox_inches="tight")
        fig4.savefig(f"{save_path}_quality.png", dpi=300, bbox_inches="tight")
        print(f"Figures saved to {save_path}_*.png")

    return figures


# Example usage
if __name__ == "__main__":
    from .cube_to_simplex import CubeToSimplexMapping, SigmoidParams

    print("Creating example mapping for visualization...")

    # Create example mapping
    sigmoid_params = [
        SigmoidParams(alpha=1.5, beta=4.0, gamma=0.3),
        SigmoidParams(alpha=1.2, beta=5.0, gamma=0.7),
    ]
    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params, special_horse_ability=0.2, noise_scale=1.0
    )

    # Create visualizations
    print("Generating comprehensive visualizations...")
    figures = visualize_mapping_comprehensive(mapping, resolution=15)

    print(f"Created {len(figures)} visualization figures.")
    print("Call plt.show() to display them.")

    # Uncomment to display:
    # plt.show()
