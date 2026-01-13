"""
Mesh building utilities for Narnia visualization.

Standalone functions for building Open3D meshes from scalar field data,
separate from the GUI viewer class.
"""

import numpy as np
import open3d as o3d
import core
from open3d.visualization import rendering


def build_mesh_from_fields(
    fields: np.ndarray,
    iso_level: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    nx: int,
    ny: int,
    total_height: float,
    z_interp_steps: int,
    smooth_method: str,
    smooth_iterations: int,
) -> o3d.geometry.TriangleMesh:
    """
    Generate mesh using marching cubes algorithm.
    
    Args:
        fields: 2D array (num_slices, nx*ny) of scalar field values
        iso_level: iso-surface threshold
        bounds_min, bounds_max: spatial bounds (XY only, Z handled by total_height)
        nx, ny: grid dimensions
        total_height: explicit Z-extent of the mesh
        z_interp_steps: number of interpolation steps between slices
        smooth_method: "none", "laplacian", "taubin", or "combined"
        smooth_iterations: number of iterations for smoothing
    
    Returns:
        Open3D TriangleMesh or None if generation fails
    """
    if fields is None or fields.shape[0] < 2:
        return None
    
    # Override Z bounds with total_height
    bounds_min_3d = np.array([bounds_min[0], bounds_min[1], 0.0])
    bounds_max_3d = np.array([bounds_max[0], bounds_max[1], total_height])
    
    # Reconstruct 3D volume from 2D slices
    grid_data = core.reconstruct_3d_volume(fields, nx, ny, bounds_min_3d, bounds_max_3d)
    if grid_data is None:
        return None
    
    volume = grid_data["volume"]
    
    # Apply Z-axis interpolation if requested
    if z_interp_steps > 0:
        volume = core.interpolate_slices(volume, z_interp_steps)
        # Recalculate spacing after interpolation
        nz_new = volume.shape[0]
        dz_new = total_height / (nz_new - 1) if nz_new > 1 else grid_data["spacing"][0]
        grid_data["spacing"] = (dz_new, grid_data["spacing"][1], grid_data["spacing"][2])
    
    # Run marching cubes
    result = core.generate_mesh_marching_cubes(
        volume,
        grid_data["spacing"],
        grid_data["origin"],
        iso_level
    )
    
    if result is None:
        return None
    
    verts, faces = result
    
    if len(verts) == 0 or len(faces) == 0:
        return None
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Apply smoothing using core.smooth_mesh
    mesh = core.smooth_mesh(mesh, method=smooth_method, iterations=smooth_iterations)
    
    return mesh


def apply_mesh_geometries(scene_widget, result_mesh, profile_mesh, bracing_mesh):
    """
    Apply mesh geometries to a scene widget with standard materials.
    
    Args:
        scene_widget: Open3D SceneWidget to add geometries to
        result_mesh: Result mesh geometry (or None)
        profile_mesh: Profile mesh geometry (or None)
        bracing_mesh: Bracing mesh geometry (or None)
    
    Returns:
        Bounding box encompassing all meshes (or None if no meshes)
    """
    bbox = None
    
    # Result mesh (typically the boolean operation result)
    if result_mesh is not None:
        r_mat = rendering.MaterialRecord()
        r_mat.shader = "defaultLit"
        r_mat.base_color = [0.9, 0.9, 0.9, 1.0]
        scene_widget.scene.add_geometry("result_mesh", result_mesh, r_mat)
        try:
            bbox = result_mesh.get_axis_aligned_bounding_box()
        except Exception:
            bbox = None
    
    # Profile mesh
    if profile_mesh is not None:
        p_mat = rendering.MaterialRecord()
        p_mat.shader = "defaultLit"
        p_mat.base_color = [0.8, 0.8, 0.8, 1.0]
        scene_widget.scene.add_geometry("profile_mesh", profile_mesh, p_mat)
        try:
            pb = profile_mesh.get_axis_aligned_bounding_box()
            if bbox is None:
                bbox = pb
            else:
                min_b = np.minimum(bbox.min_bound, pb.min_bound)
                max_b = np.maximum(bbox.max_bound, pb.max_bound)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        except Exception:
            pass

    # Bracing mesh
    if bracing_mesh is not None:
        b_mat = rendering.MaterialRecord()
        b_mat.shader = "defaultLit"
        b_mat.base_color = [0.1, 0.7, 0.95, 1.0]
        scene_widget.scene.add_geometry("bracing_mesh", bracing_mesh, b_mat)
        try:
            gb = bracing_mesh.get_axis_aligned_bounding_box()
            if bbox is None:
                bbox = gb
            else:
                min_b = np.minimum(bbox.min_bound, gb.min_bound)
                max_b = np.maximum(bbox.max_bound, gb.max_bound)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        except Exception:
            pass

    return bbox
