"""
Mesh generation from scalar fields using marching cubes algorithm.

This module provides functions for converting stacked 2D scalar field slices
into 3D triangle meshes, including interpolation, volume reconstruction, 
iso-surface extraction, and mesh post-processing.

Functions:
    interpolate_slices: Add intermediate slices via linear interpolation
    reconstruct_3d_volume: Convert 2D slices to 3D volume with spatial metadata
    generate_mesh_marching_cubes: Extract iso-surface as triangle mesh
    smooth_mesh: Apply Laplacian/Taubin smoothing filters
    export_mesh_obj: Save triangle mesh to Wavefront OBJ file

Typical workflow:
    1. interpolate_slices() - densify Z-axis sampling (optional but recommended)
    2. reconstruct_3d_volume() - reshape to 3D volume + compute spacing
    3. generate_mesh_marching_cubes() - extract iso-surface
    4. smooth_mesh() - reduce marching cubes artifacts (optional)
    5. export_mesh_obj() - save to disk

Example:
    >>> import numpy as np
    >>> from core import (interpolate_slices, reconstruct_3d_volume, 
    ...                   generate_mesh_marching_cubes, smooth_mesh, export_mesh_obj)
    >>> 
    >>> # Load scalar field data (num_slices, nx*ny)
    >>> field_data = np.load("field.npz")["result"]
    >>> bounds_min = np.array([0, 0, 0])
    >>> bounds_max = np.array([100, 100, 10])
    >>> nx, ny = 50, 50
    >>> 
    >>> # Generate mesh
    >>> interpolated = interpolate_slices(field_data, num_interpolations=2)
    >>> grid_data = reconstruct_3d_volume(interpolated, nx, ny, bounds_min, bounds_max)
    >>> result = generate_mesh_marching_cubes(grid_data["volume"], 
    ...                                        grid_data["spacing"],
    ...                                        grid_data["origin"], 
    ...                                        iso_level=0.0)
    >>> if result:
    ...     vertices, faces = result
    ...     mesh = smooth_mesh(vertices, faces, method="taubin", iterations=5)
    ...     export_mesh_obj(mesh, "output.obj")
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple
from pathlib import Path


def interpolate_slices(field_data: np.ndarray, num_interpolations: int) -> np.ndarray:
    """
    Interpolate between slices along Z-axis for smoother marching cubes results.
    
    Uses linear interpolation (scipy.ndimage.zoom) to insert additional slices
    between existing ones, improving mesh quality for datasets with sparse Z-sampling.
    
    Args:
        field_data: 3D scalar field array with shape (nz, ny, nx)
                   - nz: number of original slices
                   - ny, nx: spatial dimensions of each slice
        num_interpolations: Number of new slices to insert between each pair of
                           existing slices. Total new slices = nz + (nz-1)*num_interpolations
                           Example: 10 slices with num_interpolations=2 → 28 slices
    
    Returns:
        Interpolated 3D array with shape (new_nz, ny, nx) where:
        new_nz = nz + (nz - 1) * num_interpolations
        Returns original array unchanged if num_interpolations <= 0 or field_data is None
    
    Example:
        >>> field = np.random.rand(10, 50, 50)  # 10 slices
        >>> interpolated = interpolate_slices(field, num_interpolations=2)
        >>> interpolated.shape
        (28, 50, 50)  # 10 + (10-1)*2 = 28 slices
    
    Note:
        - Uses order=1 (linear) interpolation for speed and stability
        - Only interpolates along Z-axis; XY dimensions remain unchanged
        - Recommended num_interpolations: 1-5 for most datasets
    """
    if field_data is None or num_interpolations <= 0:
        return field_data
    
    from scipy import ndimage
    nz, ny, nx = field_data.shape
    new_nz = nz + (nz - 1) * num_interpolations
    zoom_scale = new_nz / nz
    
    # Linear interpolation along Z axis only
    interpolated = ndimage.zoom(field_data, (zoom_scale, 1, 1), order=1)
    return interpolated


def reconstruct_3d_volume(field_data_2d: np.ndarray, nx: int, ny: int, 
                          bounds_min: np.ndarray, bounds_max: np.ndarray) -> dict:
    """
    Reconstruct 3D volume from stacked 2D scalar field slices.
    
    Converts flattened 2D slice data into a proper 3D volume array with
    spatial metadata (spacing, origin) required for marching cubes meshing.
    
    Args:
        field_data_2d: 2D array with shape (num_slices, nx*ny)
                      Each row is a flattened 2D slice of the scalar field
        nx: Number of grid points in X direction (columns)
        ny: Number of grid points in Y direction (rows)
        bounds_min: Spatial bounds minimum [x_min, y_min, z_min]
        bounds_max: Spatial bounds maximum [x_max, y_max, z_max]
    
    Returns:
        Dictionary containing:
        - 'volume': 3D numpy array (num_slices, ny, nx) - reshaped scalar field
        - 'spacing': Tuple (dz, dy, dx) - voxel spacing in each dimension
        - 'origin': List [x0, y0, z0] - origin point for coordinate system
        
        Returns None if field_data_2d is None
    
    Example:
        >>> field_2d = np.random.rand(20, 2500)  # 20 slices, 50x50 grid
        >>> bounds_min = np.array([0, 0, 0])
        >>> bounds_max = np.array([100, 100, 10])
        >>> grid_data = reconstruct_3d_volume(field_2d, 50, 50, bounds_min, bounds_max)
        >>> grid_data['volume'].shape
        (20, 50, 50)
        >>> grid_data['spacing']
        (0.526..., 2.04..., 2.04...)  # (dz, dy, dx)
    
    Note:
        - Spacing is calculated as: dz = (z_max - z_min) / (num_slices - 1)
        - Spacing format (dz, dy, dx) matches scikit-image marching_cubes convention
        - Origin is set to bounds_min for proper world-space coordinates
    """
    if field_data_2d is None:
        return None
    
    num_slices = field_data_2d.shape[0]
    volume = field_data_2d.reshape((num_slices, ny, nx))
    
    # Calculate total height from Z bounds
    total_height = float(bounds_max[2] - bounds_min[2])
    dz = total_height / (num_slices - 1) if num_slices > 1 else 1.0
    
    dx = (bounds_max[0] - bounds_min[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (bounds_max[1] - bounds_min[1]) / (ny - 1) if ny > 1 else 1.0
    
    origin = [float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])]
    
    return {
        "volume": volume,
        "spacing": (dz, dy, dx),  # Z, Y, X spacing for marching cubes
        "origin": origin
    }


def generate_mesh_marching_cubes(volume: np.ndarray, spacing: Tuple[float, float, float],
                                 origin: Tuple[float, float, float], 
                                 iso_level: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate triangle mesh from 3D scalar field using marching cubes algorithm.
    
    Uses scikit-image's marching_cubes implementation to extract an iso-surface
    from a 3D volume at a specified threshold level. Handles coordinate system
    transformations to convert from scikit-image's (Z,Y,X) convention to
    standard (X,Y,Z) world coordinates.
    
    Args:
        volume: 3D scalar field array with shape (nz, ny, nx)
               Values represent the scalar field at each voxel
        spacing: Voxel spacing as (dz, dy, dx) tuple
                Controls the physical size of each voxel in world units
                Example: (0.5, 2.0, 2.0) means dz=0.5, dy=2.0, dx=2.0
        origin: World-space origin as (x0, y0, z0) tuple
               Offset to apply to all vertex coordinates
               Example: (100, 50, 0) shifts mesh to start at x=100, y=50, z=0
        iso_level: Iso-surface threshold value (default: 0.0)
                  Vertices will be placed where volume == iso_level
                  Typical range: -1.0 to 1.0 for normalized SDFs
    
    Returns:
        Tuple of (vertices, faces) where:
        - vertices: (N, 3) array of vertex positions in world coordinates (X,Y,Z)
        - faces: (M, 3) array of triangle indices (zero-based)
        
        Returns None if marching cubes fails (e.g., no iso-surface found)
    
    Example:
        >>> volume = np.random.rand(20, 50, 50)
        >>> spacing = (0.5, 2.0, 2.0)
        >>> origin = (0, 0, 0)
        >>> result = generate_mesh_marching_cubes(volume, spacing, origin, iso_level=0.5)
        >>> if result:
        ...     vertices, faces = result
        ...     print(f"Mesh: {len(vertices)} vertices, {len(faces)} triangles")
    
    Note:
        - scikit-image returns vertices in (Z,Y,X) order, this function converts to (X,Y,Z)
        - Spacing must match the order (dz, dy, dx) from reconstruct_3d_volume()
        - If no iso-surface exists at the given level, returns None
        - Typical execution time: 50-500ms depending on volume size
    
    References:
        - Lorensen & Cline (1987): "Marching Cubes: A High Resolution 3D Surface Construction Algorithm"
        - scikit-image documentation: https://scikit-image.org/docs/stable/api/skimage.measure.html#marching-cubes
    """
    try:
        from skimage import measure
        
        # Run marching cubes (returns vertices in Z,Y,X order)
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=iso_level, spacing=spacing
        )
        
        # Reorder from (z,y,x) to (x,y,z) and apply origin offset
        verts_xyz = np.zeros_like(verts)
        verts_xyz[:, 0] = verts[:, 2] + origin[0]  # X
        verts_xyz[:, 1] = verts[:, 1] + origin[1]  # Y
        verts_xyz[:, 2] = verts[:, 0] + origin[2]  # Z
        
        return verts_xyz, faces
        
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None


def smooth_mesh(mesh: "o3d.geometry.TriangleMesh", method: str = "none", 
                iterations: int = 1) -> "o3d.geometry.TriangleMesh":
    """
    Apply smoothing filters to reduce marching cubes artifacts.
    
    Smooths triangle mesh using Laplacian or Taubin filters to reduce
    staircase artifacts from marching cubes while preserving overall shape.
    
    Args:
        mesh: Open3D TriangleMesh to smooth (modified in-place)
        method: Smoothing method - one of:
               - "none": No smoothing
               - "laplacian": Laplacian smoothing (may cause shrinkage)
               - "taubin": Taubin smoothing (reduces shrinkage)
               - "combined" or "laplacian + taubin": Apply both sequentially
        iterations: Number of smoothing iterations (default: 1)
                   Typical range: 1-10 for Laplacian, 5-20 for Taubin
    
    Returns:
        Smoothed TriangleMesh (same object as input, modified in-place)
    
    Example:
        >>> import open3d as o3d
        >>> mesh = o3d.geometry.TriangleMesh.create_sphere()
        >>> smoothed = smooth_mesh(mesh, method="taubin", iterations=5)
        >>> o3d.io.write_triangle_mesh("smoothed.obj", smoothed)
    
    Note:
        - Laplacian smoothing: Fast but may shrink mesh
        - Taubin smoothing: Better volume preservation, slightly slower
        - Combined: Laplacian first, then Taubin for best quality
        - Mesh is cleaned (degenerate triangles, non-manifold edges) before smoothing
        - Normals are recomputed after smoothing
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    
    # Clean mesh topology
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    
    # Apply smoothing based on method
    method_lower = method.lower()
    if iterations > 0:
        if method_lower == "laplacian":
            mesh = mesh.filter_smooth_simple(number_of_iterations=int(iterations))
        elif method_lower == "taubin":
            mesh = mesh.filter_smooth_taubin(number_of_iterations=int(iterations))
        elif method_lower == "combined" or method_lower == "laplacian + taubin":
            # Apply both: Laplacian first, then Taubin
            mesh = mesh.filter_smooth_simple(number_of_iterations=int(iterations))
            mesh = mesh.filter_smooth_taubin(number_of_iterations=int(iterations))
    
    mesh.compute_vertex_normals()
    return mesh


def export_mesh_obj(mesh: "o3d.geometry.TriangleMesh", filepath: str, name: str = "Mesh") -> bool:
    """
    Export triangle mesh to Wavefront OBJ file format.
    
    Saves mesh vertices and faces to OBJ file with simple text format.
    Compatible with most 3D software (Rhino, Blender, Maya, etc.).
    
    Args:
        mesh: Open3D TriangleMesh to export
        filepath: Output file path (should end with .obj)
        name: Mesh name for OBJ comment header (default: "Mesh")
    
    Returns:
        True if export succeeded, False if mesh is invalid or I/O error occurred
    
    Example:
        >>> import open3d as o3d
        >>> mesh = o3d.geometry.TriangleMesh.create_box()
        >>> export_mesh_obj(mesh, "output/box.obj", name="MyBox")
        True
    
    Note:
        - OBJ format uses 1-based indexing (vertex indices start at 1)
        - Only exports geometry (vertices + faces), no materials or textures
        - Coordinates are written with 6 decimal places precision
        - File is UTF-8 encoded text
    """
    if mesh is None or len(mesh.vertices) == 0:
        return False
    
    # Ensure output directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        with open(filepath, "w") as f:
            f.write(f"# Narnia {name} Mesh\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ uses 1-based indexing
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return True
        
    except Exception as e:
        print(f"Export failed: {e}")
        return False
