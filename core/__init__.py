"""
Narnia Core API - Scalar field operations and mesh generation.

This package provides the core functionality for:
- Scalar field (SDF) operations: boolean ops, masking, morphology
- Bracing generation: Voronoi-based and keyfield blending
- Mesh generation: marching cubes from scalar fields
- Curve extraction: iso-contours from 2D slices

Modules:
    sdf_operations: Boolean operations on signed distance fields
    mesh_generator: Marching cubes mesh generation and export

Public API - Mesh Generation:
    interpolate_slices: Densify Z-axis via linear interpolation
    reconstruct_3d_volume: Convert 2D slices to 3D volume
    generate_mesh_marching_cubes: Extract iso-surface as triangle mesh
    smooth_mesh: Apply Laplacian/Taubin smoothing
    export_mesh_obj: Save mesh to OBJ file

Public API - SDF Operations:
    compute_sf_operation: Boolean ops (union, subtract, intersect)
    get_profile_mask: Apply profile bounds masking
    iso_curves_for_slice_2d: Extract contours from 2D scalar field
    generate_bracing_static: Static Voronoi bracing
    generate_bracing_keyfield_blend: Dynamic keyfield bracing
    compute_voronoi_sdf: Voronoi diagram as SDF
    postprocess_bracing_fields: Morphological cleaning

Example - Mesh generation for Grasshopper subprocess:
    >>> from narnia.core import (interpolate_slices, reconstruct_3d_volume,
    ...                          generate_mesh_marching_cubes, export_mesh_obj)
    >>> import numpy as np
    >>> 
    >>> # Load scalar field from NPZ
    >>> data = np.load("results.npz")
    >>> field = data["result"]
    >>> bounds_min = data["bounds_min"]
    >>> bounds_max = data["bounds_max"]
    >>> nx, ny = 50, 50
    >>> 
    >>> # Generate mesh
    >>> interpolated = interpolate_slices(field, num_interpolations=2)
    >>> grid_data = reconstruct_3d_volume(interpolated, nx, ny, bounds_min, bounds_max)
    >>> result = generate_mesh_marching_cubes(grid_data["volume"], 
    ...                                        grid_data["spacing"],
    ...                                        grid_data["origin"], 
    ...                                        iso_level=0.0)
    >>> if result:
    ...     vertices, faces = result
    ...     export_mesh_obj({"vertices": vertices, "triangles": faces}, "output.obj")
"""

# Mesh generation API
from .mesh_generator import (
    interpolate_slices,
    reconstruct_3d_volume,
    generate_mesh_marching_cubes,
    smooth_mesh,
    export_mesh_obj,
)

# SDF operations API
from .sdf_operations import (
    # Data loading and grid utilities
    stack_scalar_fields,
    meta_data_info,
    infer_grid_from_scalar_fields,
    
    # Boolean operations and masking
    compute_sf_operation,
    get_profile_mask,
    
    # Curve extraction
    iso_curves_for_slice_2d,
    
    # Bracing generation
    generate_bracing_static,
    generate_bracing_keyfield_blend,
    compute_voronoi_sdf,
    generate_centroids,
    constrain_centroids_to_mask,
    
    # Post-processing
    postprocess_bracing_fields,
)

__all__ = [
    # Mesh generation
    "interpolate_slices",
    "reconstruct_3d_volume",
    "generate_mesh_marching_cubes",
    "smooth_mesh",
    "export_mesh_obj",
    
    # Data loading and grid utilities
    "stack_scalar_fields",
    "meta_data_info",
    "infer_grid_from_scalar_fields",
    
    # Boolean operations and masking
    "compute_sf_operation",
    "get_profile_mask",
    
    # Curve extraction
    "iso_curves_for_slice_2d",
    
    # Bracing generation
    "generate_bracing_static",
    "generate_bracing_keyfield_blend",
    "compute_voronoi_sdf",
    "generate_centroids",
    "constrain_centroids_to_mask",
    
    # Post-processing
    "postprocess_bracing_fields",
]
