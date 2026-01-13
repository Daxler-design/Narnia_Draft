"""
python_api_demo.py — Core Package API Examples

Demonstrates how to use the core package for:
1. NPZ Workflow: Load NPZ → Generate Mesh → Export OBJ
2. JSON Workflow: Load JSON → Generate Bracing → Postprocess → Mesh

These examples show library usage for Python scripts, Jupyter notebooks,
and programmatic integration (vs CLI usage shown in grasshopper_mesh_core.py).
"""

import numpy as np
from pathlib import Path

# Core package imports
from core import (
    # Data loading
    stack_scalar_fields,
    meta_data_info,
    infer_grid_from_scalar_fields,
    
    # Mesh generation
    interpolate_slices,
    reconstruct_3d_volume,
    generate_mesh_marching_cubes,
    smooth_mesh,
    export_mesh_obj,
    
    # Bracing generation
    generate_bracing_static,
    generate_bracing_keyfield_blend,
    
    # Post-processing
    postprocess_bracing_fields,
    
    # Curves
    iso_curves_for_slice_2d,
)


# ============================================================================
# Example 1: NPZ Workflow — Load NPZ and Generate Mesh
# ============================================================================

def example_npz_workflow(npz_path, output_obj_path):
    """
    Load scalar fields from NPZ file and generate smoothed mesh.
    
    Args:
        npz_path: Path to .npz file (e.g., "output/processed_sdf_results.npz")
        output_obj_path: Path to save output mesh (e.g., "mesh_output.obj")
    """
    print("=" * 60)
    print("Example 1: NPZ Workflow")
    print("=" * 60)
    
    # 1. Load NPZ file
    print(f"\n1. Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 2. Extract field (profile_fields, bracing_fields, or result_fields)
    field_name = "profile_fields"
    field_data = data[field_name]
    
    # Reshape if flattened: (S, N) → (S, n, n)
    if field_data.ndim == 2:
        num_slices, values_per_field = field_data.shape
        n = int(np.sqrt(values_per_field))
        if n * n == values_per_field:
            field_data = field_data.reshape(num_slices, n, n)
            print(f"   Reshaped {field_name}: {num_slices} slices, {n}×{n} grid")
    
    # 3. Extract metadata
    iso_level = float(data.get("iso_level", np.array(0.0)).item())
    total_height = float(data.get("total_height", np.array(10.0)).item())
    bounds_min = data.get("bounds_min", None)
    bounds_max = data.get("bounds_max", None)
    
    # Convert 2D bounds to 3D
    if bounds_min is not None and len(bounds_min) == 2:
        bounds_min = np.array([bounds_min[0], bounds_min[1], 0.0])
    if bounds_max is not None and len(bounds_max) == 2:
        bounds_max = np.array([bounds_max[0], bounds_max[1], total_height])
    
    print(f"   Iso level: {iso_level}, Height: {total_height}")
    
    # 4. Interpolate slices for smoother mesh
    print("\n2. Interpolating slices (2x density)...")
    field_interpolated = interpolate_slices(field_data, num_interpolations=2)
    print(f"   {field_data.shape[0]} slices → {field_interpolated.shape[0]} slices")
    
    # 5. Reconstruct 3D volume
    print("\n3. Reconstructing 3D volume...")
    nz, ny, nx = field_interpolated.shape
    grid_data = reconstruct_3d_volume(
        field_interpolated,
        nx=nx,
        ny=ny,
        bounds_min=bounds_min,
        bounds_max=bounds_max
    )
    print(f"   Volume shape: {grid_data['volume'].shape}")
    print(f"   Spacing: {grid_data['spacing']}")
    print(f"   Origin: {grid_data['origin']}")
    
    # 6. Generate mesh via marching cubes
    print("\n4. Generating mesh (marching cubes)...")
    result = generate_mesh_marching_cubes(
        grid_data["volume"],
        grid_data["spacing"],
        grid_data["origin"],
        iso_level=iso_level
    )
    
    if result is None:
        print("   ❌ No geometry generated (iso level outside data range)")
        return
    
    vertices, faces = result
    print(f"   ✓ Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # 7. Apply smoothing (optional)
    print("\n5. Smoothing mesh (Taubin filter)...")
    mesh_dict = {"vertices": vertices, "triangles": faces}
    smoothed = smooth_mesh(mesh_dict, method="taubin", iterations=5)
    
    if smoothed:
        vertices = smoothed["vertices"]
        faces = smoothed["triangles"]
        print("   ✓ Smoothing applied")
    else:
        print("   ⚠ Smoothing skipped (Open3D not available)")
    
    # 8. Export to OBJ
    print(f"\n6. Exporting mesh to: {output_obj_path}")
    export_mesh_obj({"vertices": vertices, "triangles": faces}, output_obj_path)
    print("   ✓ Export complete\n")


# ============================================================================
# Example 2: JSON Workflow — Load JSON, Generate Bracing, Postprocess, Mesh
# ============================================================================

def example_json_workflow(json_path, output_npz_path):
    """
    Load scalar fields from JSON, generate bracing, postprocess, and save NPZ.
    
    Args:
        json_path: Path to JSON file with scalar_field_values_0, scalar_field_values_1, ...
        output_npz_path: Path to save processed fields (e.g., "processed_output.npz")
    """
    print("=" * 60)
    print("Example 2: JSON Workflow — Bracing Generation")
    print("=" * 60)
    
    # 1. Load JSON data
    print(f"\n1. Loading JSON: {json_path}")
    import json
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    # 2. Stack scalar fields
    print("\n2. Stacking scalar fields...")
    profile_fields_2d, field_keys = stack_scalar_fields(data_dict, prefix="scalar_field_values_")
    print(f"   Loaded {len(field_keys)} fields")
    
    # 3. Extract metadata
    iso_level, slice_count, bounds_max, bounds_min = meta_data_info(data_dict)
    print(f"   Iso level: {iso_level}, Slices: {slice_count}")
    
    # 4. Infer grid dimensions
    num_fields, nx, ny = infer_grid_from_scalar_fields(profile_fields_2d)
    print(f"   Grid: {num_fields} slices × {nx}×{ny}")
    
    # 5. Generate static Voronoi bracing
    print("\n3. Generating static Voronoi bracing (k=5 centroids)...")
    bracing_fields = generate_bracing_static(
        profile_fields_2d,
        iso_level=iso_level or 0.0,
        nx=nx,
        ny=ny,
        k=5,
        seed=42
    )
    print(f"   ✓ Generated bracing fields: {bracing_fields.shape}")
    
    # 6. Postprocess bracing (morphological cleaning + temporal smoothing)
    print("\n4. Post-processing bracing fields...")
    cleaned_bracing = postprocess_bracing_fields(
        bracing_fields,
        profile_fields_2d,
        iso_profile=iso_level or 0.0,
        iso_brace=0.0,
        close_radius=2,
        min_area=120,
        temporal_window=3
    )
    print(f"   ✓ Cleaned bracing fields: {cleaned_bracing.shape}")
    
    # 7. Save to NPZ for later mesh generation
    print(f"\n5. Saving to NPZ: {output_npz_path}")
    np.savez_compressed(
        output_npz_path,
        profile_fields=profile_fields_2d,
        bracing_fields=cleaned_bracing,
        iso_level=iso_level,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        total_height=data_dict.get("total_height", 10.0)
    )
    print("   ✓ Saved NPZ with profile_fields and bracing_fields\n")


# ============================================================================
# Example 3: Keyfield Blending Workflow
# ============================================================================

def example_keyfield_workflow(profile_fields_2d, iso_level, nx, ny):
    """
    Generate bracing with keyfield blending (variable centroids per slice).
    
    Args:
        profile_fields_2d: Profile scalar fields (num_slices, nx*ny)
        iso_level: Iso level for profile mask
        nx, ny: Grid dimensions
    
    Returns:
        Blended bracing fields (num_slices, nx*ny)
    """
    print("=" * 60)
    print("Example 3: Keyfield Blending Workflow")
    print("=" * 60)
    
    # Define key slices with varying centroid counts
    # Format: [(slice_idx, num_centroids), ...]
    num_slices = profile_fields_2d.shape[0]
    keys_config = [
        (0, 3),                    # Start: 3 centroids
        (num_slices // 2, 5),      # Middle: 5 centroids
        (num_slices - 1, 4)        # End: 4 centroids
    ]
    
    print(f"\n1. Key slices configuration:")
    for slice_idx, k in keys_config:
        print(f"   Slice {slice_idx}: {k} centroids")
    
    # Generate bracing with Ridge Response blending
    print("\n2. Generating bracing with keyfield blending...")
    print("   Using Ridge Response: R = exp(-(V/σ)²)")
    
    bracing_fields = generate_bracing_keyfield_blend(
        profile_fields_2d,
        iso_level=iso_level,
        nx=nx,
        ny=ny,
        keys_config=keys_config,
        smooth=0.5,      # 50% smoothstep interpolation
        sigma=5.0,       # Ridge width parameter
        tau=0.5,         # Ridge threshold offset
        beta=4.0,        # Exponential blend softness
        seed=42
    )
    
    print(f"   ✓ Generated blended bracing: {bracing_fields.shape}\n")
    return bracing_fields


# ============================================================================
# Example 4: Extract ISO-Curves for 2D Visualization
# ============================================================================

def example_iso_curves(npz_path, slice_idx=30):
    """
    Extract and visualize iso-curves from a specific slice.
    
    Args:
        npz_path: Path to NPZ file
        slice_idx: Slice index to extract curves from
    """
    print("=" * 60)
    print("Example 4: ISO-Curve Extraction")
    print("=" * 60)
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    field_data = data["profile_fields"]
    
    # Reshape if needed
    if field_data.ndim == 2:
        num_slices, values_per_field = field_data.shape
        n = int(np.sqrt(values_per_field))
        field_data = field_data.reshape(num_slices, n, n)
    
    # Get slice
    slice_2d = field_data[slice_idx]
    ny, nx = slice_2d.shape
    
    # Create coordinate grids
    bounds_min = data.get("bounds_min", np.array([0, 0]))
    bounds_max = data.get("bounds_max", np.array([10, 10]))
    
    x = np.linspace(bounds_min[0], bounds_max[0], nx)
    y = np.linspace(bounds_min[1], bounds_max[1], ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Extract curves at iso_level=0.0
    print(f"\n1. Extracting curves from slice {slice_idx}/{len(field_data)-1}")
    curves = iso_curves_for_slice_2d(slice_2d, level=0.0, X=X, Y=Y)
    
    print(f"   ✓ Found {len(curves)} iso-curve segments")
    for i, curve in enumerate(curves):
        print(f"      Curve {i}: {len(curve)} points")
    
    print("\n   Use these curves for 2D preview in GH/CAD software\n")


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Narnia Core Package — Python API Demonstration")
    print("=" * 60 + "\n")
    
    # Check for example NPZ file
    npz_files = list(Path("output").glob("*.npz"))
    
    if npz_files:
        example_npz = str(npz_files[0])
        
        # Run Example 1: NPZ → Mesh
        example_npz_workflow(
            npz_path=example_npz,
            output_obj_path="examples/demo_mesh.obj"
        )
        
        # Run Example 4: ISO-Curves
        example_iso_curves(example_npz, slice_idx=30)
        
    else:
        print("⚠ No NPZ files found in output/ directory")
        print("  Skipping NPZ-based examples\n")
    
    # Check for example JSON file
    json_files = list(Path("alice_result").rglob("*.json"))
    
    if json_files:
        # Filter for files with scalar field data
        for json_path in json_files:
            if "waveStackFields" in json_path.name:
                # Run Example 2: JSON → Bracing → NPZ
                example_json_workflow(
                    json_path=str(json_path),
                    output_npz_path="examples/demo_bracing.npz"
                )
                break
    else:
        print("⚠ No JSON files found in alice_result/ directory")
        print("  Skipping JSON-based examples\n")
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - examples/demo_mesh.obj (if NPZ available)")
    print("  - examples/demo_bracing.npz (if JSON available)")
    print("\nSee examples/grasshopper_mesh_core.py for CLI usage\n")
