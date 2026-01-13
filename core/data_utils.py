"""
Data loading and grid utilities for scalar fields.

Functions for loading scalar field data from JSON, inferring grid dimensions,
and extracting metadata.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any


def stack_scalar_fields(data_dict: Dict[str, Any], prefix: str = "scalar_field_values_") -> Tuple[np.ndarray, List[str]]:
    """
    Extract all scalar field keys from a JSON dict and stack them
    into a 2D NumPy array, sorted by their numeric index.
    
    Args:
        data_dict: Dictionary containing scalar field data
        prefix: Prefix for scalar field keys (default: "scalar_field_values_")
    
    Returns:
        Tuple of (stacked_fields, field_keys) where:
        - stacked_fields: 2D array (num_slices, values_per_slice)
        - field_keys: List of sorted field key names
    
    Example:
        >>> data = {"scalar_field_values_0": [...], "scalar_field_values_1": [...]}
        >>> fields, keys = stack_scalar_fields(data)
        >>> fields.shape
        (2, 2500)  # 2 slices, 50x50 grid
    """
    scalar_field_keys = sorted(
        [key for key in data_dict.keys() if key.startswith(prefix)],
        key=lambda x: int(x.split('_')[-1])
    )
    scalar_fields_2d = np.array([data_dict[key] for key in scalar_field_keys])
    return scalar_fields_2d, scalar_field_keys


def meta_data_info(data_dict: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], Any, Any]:
    """
    Extract metadata information from the JSON dict.
    
    Args:
        data_dict: Dictionary containing metadata fields
    
    Returns:
        Tuple of (iso_level, slice_count, bounds_max, bounds_min)
    
    Example:
        >>> data = {"iso_level": 0.0, "slice_count": 60, "bounds_min": [0,0,0], "bounds_max": [100,100,10]}
        >>> iso, count, bmax, bmin = meta_data_info(data)
        >>> iso
        0.0
    """
    bounds_max = data_dict.get("bounds_max", None)
    bounds_min = data_dict.get("bounds_min", None)
    iso_level = data_dict.get("iso_level", None) 
    slice_count = data_dict.get("slice_count", None)
    total_height = data_dict.get("total_height", None)
    return iso_level, slice_count, bounds_max, bounds_min


def infer_grid_from_scalar_fields(scalar_fields_2d: np.ndarray) -> Tuple[int, int, int]:
    """
    Infer (num_slices, nx, ny) from scalar_fields_2d assuming a square grid.
    
    Args:
        scalar_fields_2d: 2D array with shape (num_slices, nx*ny)
    
    Returns:
        Tuple of (num_slices, nx, ny) where nx == ny for square grids
    
    Raises:
        ValueError: If values_per_field is not a perfect square
    
    Example:
        >>> fields = np.random.rand(20, 2500)  # 20 slices, 50x50 grid
        >>> num_slices, nx, ny = infer_grid_from_scalar_fields(fields)
        >>> (num_slices, nx, ny)
        (20, 50, 50)
    """
    num_fields, values_per_field = scalar_fields_2d.shape
    n = int(np.sqrt(values_per_field))
    if n * n != values_per_field:
        raise ValueError(f"Cannot infer square grid from {values_per_field} values.")
    return num_fields, n, n
