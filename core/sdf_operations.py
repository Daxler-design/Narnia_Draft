"""
Core scalar field operations for boolean-like SDF operations.

Provides compute_sf_operation for union/intersection/difference operations on SDFs.
All other functions have been extracted to specialized modules:
- data_utils: stack_scalar_fields, meta_data_info, infer_grid_from_scalar_fields
- curves: iso_curves_for_slice_2d
- bracing_generator: generate_bracing_*, compute_voronoi_sdf, generate_centroids, constrain_centroids_to_mask
- postprocess: postprocess_bracing_fields, _binary_dilate, _binary_erode, _label_components
"""

import numpy as np


def compute_sf_operation(
    sf_A: np.ndarray,
    sf_B: np.ndarray,
    iso_level_A: float = 0.0,
    iso_level_B: float = 0.0,
    mode: str = "difference",
    swap: bool = False
) -> np.ndarray:
    """
    Compute an SDF boolean-like operation between two scalar fields.
    
    Args:
        sf_A: First scalar field array
        sf_B: Second scalar field array
        iso_level_A: Iso level for field A (default: 0.0)
        iso_level_B: Iso level for field B (default: 0.0)
        mode: Operation mode - 'difference'/'union'/'intersection' (default: 'difference')
        swap: If True, swap A and B before operation (default: False)
    
    Returns:
        Result scalar field after boolean operation
    
    Modes:
        - 'difference' / 'a_minus_b' / 'sub': max(A, -B)
        - 'union' / 'or' / 'min': min(A, B)
        - 'intersection' / 'and' / 'max': max(A, B)
    
    Example:
        >>> A = np.random.rand(100, 100) - 0.5
        >>> B = np.random.rand(100, 100) - 0.5
        >>> result = compute_sf_operation(A, B, mode='difference')
        >>> result.shape
        (100, 100)
    
    Note:
        - Fields are shifted by iso levels before operation
        - All modes assume SDF convention (negative inside, positive outside)
    """
    A = np.asarray(sf_A, dtype=float) - iso_level_A
    B = np.asarray(sf_B, dtype=float) - iso_level_B

    if A.shape != B.shape:
        raise ValueError("Input scalar fields must have the same shape")

    if swap:
        A, B = B, A

    m = mode.lower()
    if m in ("difference", "a_minus_b", "sub"):
        result = np.maximum(A, -B)
    elif m in ("union", "or", "min"):
        result = np.minimum(A, B)
    elif m in ("intersection", "and", "max"):
        result = np.maximum(A, B)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return result
