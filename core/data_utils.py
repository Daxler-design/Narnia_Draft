"""
Data loading and grid utilities for scalar fields.

Functions for loading scalar field data from JSON, inferring grid dimensions,
and extracting metadata.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt


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


def poly_to_true_sdf(poly_xyz, bbox_min, bbox_max, nx=256, ny=256):
    """
    Convert a closed polyline to a true signed distance field using Euclidean Distance Transform.
    
    Creates an accurate SDF with world-unit distances where negative values represent
    the interior of the polygon and positive values represent the exterior.
    
    Args:
        poly_xyz: Polyline points array (m, 3) - will use only x,y coordinates
                 Automatically closed if first != last point
        bbox_min: Bounding box minimum [x, y, z] in world units
        bbox_max: Bounding box maximum [x, y, z] in world units
        nx: Grid width (default: 256)
        ny: Grid height (default: 256)
    
    Returns:
        2D SDF array (ny, nx) with world-unit distances
        - Negative values: inside polygon (material)
        - Positive values: outside polygon (void)
        - Zero: on polygon boundary surface
        - |∇SDF| ≈ 1 (true distance property)
    
    Example:
        >>> poly = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
        >>> bbox_min, bbox_max = [0, 0, 0], [10, 10, 10]
        >>> sdf = poly_to_true_sdf(poly, bbox_min, bbox_max, nx=128, ny=128)
        >>> sdf.shape
        (128, 128)
        >>> np.min(sdf) < 0  # Has interior
        True
    
    Note:
        - Uses matplotlib.path for point-in-polygon test
        - Uses scipy.ndimage.distance_transform_edt with world-unit sampling
        - Automatically handles unclosed polylines
        - Grid spacing accounts for world-unit bbox for accurate distances
    """
    poly_xyz = np.asarray(poly_xyz, dtype=float)
    poly_xy = poly_xyz[:, :2]

    # Ensure closed polyline
    if np.linalg.norm(poly_xy[0] - poly_xy[-1]) > 1e-9:
        poly_xy = np.vstack([poly_xy, poly_xy[0:1]])

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Inside mask via point-in-polygon
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    inside = MplPath(poly_xy).contains_points(pts).reshape(ny, nx)

    # EDT (true distance in world units because sampling=(dy,dx))
    dist_in = distance_transform_edt(inside, sampling=(dy, dx))
    dist_out = distance_transform_edt(~inside, sampling=(dy, dx))

    # Signed: inside negative, outside positive
    phi = dist_out - dist_in
    return phi


def load_sdf_list_from_inshapes(json_path, nx=256, ny=256, branch_index=0):
    """
    Load polylines from inShapes.json and convert to true SDF list.
    
    Reads polygon data from JSON file containing shape definitions and bounding box,
    then converts each polyline to a true signed distance field.
    
    Args:
        json_path: Path to inShapes.json file containing polyline data
        nx: Grid width for SDF (default: 256)
        ny: Grid height for SDF (default: 256)
        branch_index: Which shape branch to load (default: 0)
    
    Returns:
        List of SDF arrays, one per slice/polyline
        Each SDF is (ny, nx) with world-unit signed distances
    
    Example:
        >>> sdfs = load_sdf_list_from_inshapes("alice_result/inShapes.json", nx=256, ny=256)
        >>> len(sdfs)
        60  # Number of slices
        >>> sdfs[0].shape
        (256, 256)
    
    Note:
        - Requires inShapes.json format: {"shapes": [...], "bbox": {"minbb": [...], "maxbb": [...]}}
        - Each shape contains "polys" list of polylines
        - All SDFs share same bounding box from JSON
        - Progress printed for large datasets
    """
    data = json.loads(Path(json_path).read_text())
    bbox_min = data["bbox"]["minbb"]
    bbox_max = data["bbox"]["maxbb"]

    polys = data["shapes"][branch_index]["polys"]
    sdfs = [poly_to_true_sdf(poly, bbox_min, bbox_max, nx=nx, ny=ny) for poly in polys]
    return sdfs
