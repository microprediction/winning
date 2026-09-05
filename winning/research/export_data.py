"""
Export diffeomorphism mapping data to JSON for JavaScript visualizations.

This module provides functions to export mapping data, quality metrics,
and configuration parameters in JSON format for interactive web visualizations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from .cube_to_simplex import CubeToSimplexMapping, SigmoidParams

try:
    from .adaptive_special_horse import AdaptiveSpecialHorse, SpecialHorseConfig
    from .enhanced_cube_to_simplex import EnhancedCubeToSimplexMapping
    from .quality_assessment import comprehensive_quality_assessment

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def numpy_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_native(item) for item in obj]
    else:
        return obj


def export_mapping_data(
    mapping,
    resolution: int = 20,
    include_quality: bool = True,
    quality_samples: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Export mapping data to JSON-serializable format.

    Args:
        mapping: The diffeomorphism mapping to export
        resolution: Grid resolution for cube points
        include_quality: Whether to compute quality metrics
        quality_samples: Sample sizes for quality assessment

    Returns:
        Dictionary with mapping data ready for JSON export
    """
    # Generate grid of cube points
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])

    # Map to simplex
    simplex_points = []
    for point in cube_points:
        simplex_point = mapping(point)
        simplex_points.append(simplex_point)

    simplex_points = np.array(simplex_points)

    # Prepare export data
    export_data = {
        "metadata": {
            "resolution": resolution,
            "k_dimension": mapping.k,
            "total_points": len(cube_points),
            "export_timestamp": np.datetime64("now").item().isoformat(),
        },
        "mapping_data": {
            "cube_points": cube_points.tolist(),
            "simplex_points": simplex_points.tolist(),
        },
        "configuration": {},
    }

    # Add configuration based on mapping type
    if hasattr(mapping, "sigmoid_params"):
        # Standard or enhanced mapping
        sigmoid_config = []
        for param in mapping.sigmoid_params:
            if isinstance(param, SigmoidParams):
                sigmoid_config.append(
                    {"alpha": param.alpha, "beta": param.beta, "gamma": param.gamma}
                )
            else:
                sigmoid_config.append(param)
        export_data["configuration"]["sigmoid_params"] = sigmoid_config

        if hasattr(mapping, "special_horse_ability"):
            export_data["configuration"]["special_horse_ability"] = mapping.special_horse_ability
        elif hasattr(mapping, "special_horse") and ENHANCED_AVAILABLE:
            # Enhanced mapping with adaptive special horse
            config = mapping.special_horse.config
            export_data["configuration"]["special_horse"] = {
                "distribution": (
                    config.distribution.value
                    if hasattr(config.distribution, "value")
                    else str(config.distribution)
                ),
                "base_ability": config.base_ability,
                "location": config.location,
                "scale": config.scale,
                "shape": config.shape,
                "adaptive_mode": config.adaptive_mode,
            }

    # Add quality metrics if requested
    if include_quality:
        try:
            if quality_samples is None:
                quality_samples = {
                    "symmetry_samples": 100,
                    "volume_samples": 20,
                    "smoothness_samples": 20,
                    "coverage_samples": 100,
                    "invertibility_samples": 10,
                }

            metrics = comprehensive_quality_assessment(mapping, **quality_samples)

            export_data["quality_metrics"] = {
                "symmetry_score": metrics.symmetry_score,
                "volume_preservation_score": metrics.volume_preservation_score,
                "smoothness_score": metrics.smoothness_score,
                "coverage_score": metrics.coverage_score,
                "invertibility_score": metrics.invertibility_score,
                "overall_score": metrics.overall_score({}),
            }
        except Exception as e:
            export_data["quality_metrics"] = {"error": str(e)}

    # Convert numpy types to native Python types
    return numpy_to_native(export_data)


def export_mapping_to_file(
    mapping, filename: str, resolution: int = 20, include_quality: bool = True
) -> str:
    """
    Export mapping data to JSON file.

    Args:
        mapping: The diffeomorphism mapping to export
        filename: Output filename (should end with .json)
        resolution: Grid resolution for cube points
        include_quality: Whether to compute quality metrics

    Returns:
        Path to exported file
    """
    data = export_mapping_data(mapping, resolution, include_quality)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename


def export_parameter_study_data(
    parameter_configs: List[Dict[str, Any]],
    resolution: int = 10,
    output_file: str = "parameter_study.json",
) -> str:
    """
    Export data for multiple parameter configurations for comparative visualization.

    Args:
        parameter_configs: List of parameter configurations to test
        resolution: Grid resolution for each mapping
        output_file: Output filename

    Returns:
        Path to exported file
    """
    study_data = {
        "metadata": {
            "total_configurations": len(parameter_configs),
            "resolution": resolution,
            "export_timestamp": np.datetime64("now").item().isoformat(),
        },
        "configurations": [],
    }

    for i, config in enumerate(parameter_configs):
        print(f"Processing configuration {i + 1}/{len(parameter_configs)}")

        # Create mapping from configuration
        sigmoid_params = []
        for j in range(len(config.get("sigmoid_params", []))):
            param_dict = config["sigmoid_params"][j]
            sigmoid_params.append(SigmoidParams(**param_dict))

        if ENHANCED_AVAILABLE and "special_horse" in config:
            # Enhanced mapping
            from .adaptive_special_horse import (
                AdaptiveSpecialHorse,
                DistributionType,
                SpecialHorseConfig,
            )

            special_config_dict = config["special_horse"].copy()
            if "distribution" in special_config_dict and isinstance(
                special_config_dict["distribution"], str
            ):
                # Convert string to enum
                special_config_dict["distribution"] = DistributionType(
                    special_config_dict["distribution"]
                )

            special_config = SpecialHorseConfig(**special_config_dict)
            special_horse = AdaptiveSpecialHorse(special_config)
            mapping = EnhancedCubeToSimplexMapping(
                sigmoid_params=sigmoid_params, special_horse=special_horse
            )
        else:
            # Standard mapping
            special_ability = config.get("special_horse_ability", 0.0)
            mapping = CubeToSimplexMapping(
                sigmoid_params=sigmoid_params, special_horse_ability=special_ability
            )

        # Export this configuration
        config_data = export_mapping_data(
            mapping,
            resolution,
            include_quality=True,
            quality_samples={
                "symmetry_samples": 50,
                "volume_samples": 10,
                "smoothness_samples": 10,
                "coverage_samples": 50,
                "invertibility_samples": 5,
            },
        )
        config_data["config_index"] = i
        config_data["config_name"] = config.get("name", f"Config_{i}")

        study_data["configurations"].append(config_data)

    with open(output_file, "w") as f:
        json.dump(study_data, f, indent=2)

    print(f"Exported {len(parameter_configs)} configurations to {output_file}")
    return output_file


# Example usage functions
def create_example_exports():
    """Create example data exports for visualization development."""

    # Example 1: Simple symmetric mapping
    symmetric_params = [
        SigmoidParams(alpha=1.2, beta=4.0, gamma=0.5),
        SigmoidParams(alpha=1.2, beta=4.0, gamma=0.5),
    ]
    symmetric_mapping = CubeToSimplexMapping(
        sigmoid_params=symmetric_params, special_horse_ability=0.0
    )
    export_mapping_to_file(symmetric_mapping, "docs/interactive/data/symmetric_mapping.json")

    # Example 2: Asymmetric mapping
    asymmetric_params = [
        SigmoidParams(alpha=1.5, beta=6.0, gamma=0.3),
        SigmoidParams(alpha=0.8, beta=3.0, gamma=0.7),
    ]
    asymmetric_mapping = CubeToSimplexMapping(
        sigmoid_params=asymmetric_params, special_horse_ability=0.4
    )
    export_mapping_to_file(asymmetric_mapping, "docs/interactive/data/asymmetric_mapping.json")

    if ENHANCED_AVAILABLE:
        # Example 3: Enhanced mapping with adaptive special horse
        from .adaptive_special_horse import (
            AdaptiveSpecialHorse,
            DistributionType,
            SpecialHorseConfig,
        )

        enhanced_config = SpecialHorseConfig(
            distribution=DistributionType.NORMAL,
            base_ability=0.5,
            location=0.0,
            scale=1.2,
            adaptive_mode="mean_adaptive",
        )
        enhanced_horse = AdaptiveSpecialHorse(enhanced_config)
        enhanced_mapping = EnhancedCubeToSimplexMapping(
            sigmoid_params=symmetric_params, special_horse=enhanced_horse
        )
        export_mapping_to_file(enhanced_mapping, "docs/interactive/data/enhanced_mapping.json")

    print("Example exports created in docs/interactive/data/")


if __name__ == "__main__":
    import os

    # Create data directory if it doesn't exist
    os.makedirs("docs/interactive/data", exist_ok=True)

    # Create example exports
    create_example_exports()
