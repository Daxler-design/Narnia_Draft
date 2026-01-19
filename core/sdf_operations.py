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
import debug_utils as debug
    

def compute_sf_operation(
    sf_A: np.ndarray,
    sf_B: np.ndarray,
    iso_level_A: float = 0.0,
    iso_level_B: float = 0.0,
    mode: str = "difference",
    swap: bool = False
) -> np.ndarray:
    """
    Compute boolean-like operations between two signed distance fields (SDFs).
    
    Performs CSG-style operations (union, intersection) and custom difference on scalar fields
    using SDF convention where negative values represent interior and positive values
    represent exterior. The iso-surface (boundary) occurs at value = 0.
    
    Args:
        sf_A: First scalar field array (any shape)
               SDF convention: negative = inside, positive = outside
        sf_B: Second scalar field array (must match sf_A shape)
               SDF convention: negative = inside, positive = outside
        iso_level_A: Iso level for field A (default: 0.0)
                    Values are shifted by this amount: A' = A - iso_level_A
        iso_level_B: Iso level for field B (default: 0.0)
                    Values are shifted by this amount: B' = B - iso_level_B
        mode: Operation mode (case-insensitive):
             - 'difference' / 'a_minus_b' / 'sub': Subtract offset fields
             - 'union' / 'or' / 'min': Combine A and B (largest volume)
             - 'intersection' / 'and' / 'max': Overlap of A and B
        swap: If True, swap A and B before operation (default: False)
              Useful for computing "B minus A" with mode='difference'
    
    Returns:
        Result scalar field with same shape as inputs
        Output follows SDF convention (negative inside, positive outside)
    
    Operation Details:
        After shifting by iso_levels, the operations are:
        
        1. Difference (A - B):
           result = sf_A - ((sf_A - iso_level_A) - sf_B)
                  = iso_level_A + sf_B
           - Custom formula: result depends on iso_level_A and bracing field only
           - Does NOT use iso_level_B for bracing in difference mode
           - Result is independent of actual profile SDF values
           
        2. Union (A ∪ B):
           result = min(A - iso_level_A, B - iso_level_B)
           - Standard CSG union with both offsets applied
           - Combines both volumes
           
        3. Intersection (A ∩ B):
           result = max(A - iso_level_A, B - iso_level_B)
           - Standard CSG intersection with both offsets applied
           - Only keeps overlapping region
    
    Examples:
        >>> # Example 1: Subtract bracing from profile
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> bracing = np.random.rand(60, 2500) - 0.3
        >>> result = compute_sf_operation(profile, bracing, 
        ...                               iso_level_A=0.0, 
        ...                               iso_level_B=0.0,
        ...                               mode='difference')
        >>> result.shape
        (60, 2500)
        
        >>> # Example 2: Union of two overlapping volumes
        >>> sphere_A = create_sphere_sdf(center=[0, 0, 0], radius=5)
        >>> sphere_B = create_sphere_sdf(center=[3, 0, 0], radius=5)
        >>> combined = compute_sf_operation(sphere_A, sphere_B, mode='union')
        
        >>> # Example 3: Intersection (boolean AND)
        >>> intersection = compute_sf_operation(sphere_A, sphere_B, mode='intersection')
        
        >>> # Example 4: Swap for "B minus A"
        >>> cavity = compute_sf_operation(sphere_A, sphere_B, 
        ...                               mode='difference', swap=True)
    
    Note:
        - Input fields are shifted by iso_levels BEFORE operation
        - All modes preserve SDF properties (signed distance approximation)
        - For best results, inputs should be true SDFs (distance to surface)
        - Operation is performed element-wise (no spatial coupling)
        - Output range depends on input ranges and operation type
    
    Raises:
        ValueError: If sf_A and sf_B have different shapes
        ValueError: If mode is not recognized
    """
    # Apply iso-level offsets to both fields
    A = np.asarray(sf_A, dtype=float) - iso_level_A
    B = np.asarray(sf_B, dtype=float) - iso_level_B



    if sf_A.shape != sf_B.shape:
        raise ValueError("Input scalar fields must have the same shape")

    # Handle swap
    if swap:
        A, B = B, A

    m = mode.lower()
    if m in ("difference", "a_minus_b", "sub"):
        # Standard CSG difference: A - B
        # Points inside A (A<0) but outside B (B>0) remain
        # Points inside both get removed
        result = np.maximum(A, -B)
        
        # Debug output
        num_vals = sf_A.shape[-1]
        nx = ny = int(np.sqrt(num_vals))
        debug._save_debug_A_offsetA_B(sf_A, A, result, nx, ny, slice_idx=20, tag=mode)
    elif m in ("union", "or", "min"):
        # Standard CSG union: A ∪ B
        # Interior where either A or B is inside
        result = np.minimum(A, B)
    elif m in ("intersection", "and", "max"):
        # Standard CSG intersection: A ∩ B
        # Interior only where both A and B are inside
        result = np.maximum(A, B)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return result


def compute_sf_operation_OLD(
    sf_A: np.ndarray,
    sf_B: np.ndarray,
    iso_level_A: float = 0.0,
    iso_level_B: float = 0.0,
    mode: str = "difference",
    swap: bool = False
) -> np.ndarray:
    """
    Compute boolean-like operations between two signed distance fields (SDFs).
    
    Performs CSG-style operations (union, intersection, difference) on scalar fields
    using SDF convention where negative values represent interior and positive values
    represent exterior. The iso-surface (boundary) occurs at value = 0.
    
    Args:
        sf_A: First scalar field array (any shape)
               SDF convention: negative = inside, positive = outside
        sf_B: Second scalar field array (must match sf_A shape)
               SDF convention: negative = inside, positive = outside
        iso_level_A: Iso level for field A (default: 0.0)
                    Values are shifted by this amount: A' = A - iso_level_A
        iso_level_B: Iso level for field B (default: 0.0)
                    Values are shifted by this amount: B' = B - iso_level_B
        mode: Operation mode (case-insensitive):
             - 'difference' / 'a_minus_b' / 'sub': Subtract B from A
             - 'union' / 'or' / 'min': Combine A and B (largest volume)
             - 'intersection' / 'and' / 'max': Overlap of A and B
        swap: If True, swap A and B before operation (default: False)
              Useful for computing "B minus A" with mode='difference'
    
    Returns:
        Result scalar field with same shape as inputs
        Output follows SDF convention (negative inside, positive outside)
    
    Operation Details:
        After shifting by iso_levels, the operations are:
        
        1. Difference (A - B):
           result = max(A, -B)
           - Interior where A is inside AND B is outside
           - Creates cavity by subtracting B's interior from A
           
        2. Union (A ∪ B):
           result = min(A, B)
           - Interior where A OR B is inside
           - Combines both volumes
           
        3. Intersection (A ∩ B):
           result = max(A, B)
           - Interior where A AND B are both inside
           - Only keeps overlapping region
    
    Examples:
        >>> # Example 1: Subtract bracing from profile
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> bracing = np.random.rand(60, 2500) - 0.3
        >>> result = compute_sf_operation(profile, bracing, 
        ...                               iso_level_A=0.0, 
        ...                               iso_level_B=0.0,
        ...                               mode='difference')
        >>> result.shape
        (60, 2500)
        
        >>> # Example 2: Union of two overlapping volumes
        >>> sphere_A = create_sphere_sdf(center=[0, 0, 0], radius=5)
        >>> sphere_B = create_sphere_sdf(center=[3, 0, 0], radius=5)
        >>> combined = compute_sf_operation(sphere_A, sphere_B, mode='union')
        
        >>> # Example 3: Intersection (boolean AND)
        >>> intersection = compute_sf_operation(sphere_A, sphere_B, mode='intersection')
        
        >>> # Example 4: Swap for "B minus A"
        >>> cavity = compute_sf_operation(sphere_A, sphere_B, 
        ...                               mode='difference', swap=True)
    
    Note:
        - Input fields are shifted by iso_levels BEFORE operation
        - All modes preserve SDF properties (signed distance approximation)
        - For best results, inputs should be true SDFs (distance to surface)
        - Operation is performed element-wise (no spatial coupling)
        - Output range depends on input ranges and operation type
    
    Raises:
        ValueError: If sf_A and sf_B have different shapes
        ValueError: If mode is not recognized
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
