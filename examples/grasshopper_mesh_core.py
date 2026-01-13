"""
grasshopper_mesh_core.py — Enhanced MeshFromNPZ using core package

This is an EXPERIMENTAL enhanced version of MeshFromNPZ.py that uses the refactored
core package. MeshFromNPZ.py remains the stable, production-proven tool.

Enhancements:
- Uses core package functions (interpolate_slices, reconstruct_3d_volume, etc.)
- Adds --smooth flag for Laplacian/Taubin/Combined mesh smoothing
- Maintains identical CLI interface for drop-in testing

CLI Usage (identical to MeshFromNPZ.py):
    python grasshopper_mesh_core.py --npz data.npz --field profile_fields --iso 0.0

New Features:
    python grasshopper_mesh_core.py --npz data.npz --smooth laplacian --iterations 5

Grasshopper Integration:
    Use the same subprocess pattern as MeshFromNPZ.py (see README.md)
    Test this enhanced version by swapping ScriptPath in your GH component
"""

import numpy as np
import os
import argparse
import sys

# Add parent directory to path for core package import (needed when running from examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import core package functions
try:
    from core import interpolate_slices, reconstruct_3d_volume, generate_mesh_marching_cubes, smooth_mesh
except ImportError:
    sys.stderr.write("Error: Cannot import core package. Ensure core/ is in PYTHONPATH.\n")
    sys.exit(1)


def load_npz_file(file_path, verbose=False):
    """Load NPZ file and optionally print structure."""
    data = np.load(file_path, allow_pickle=True)
    if verbose:
        print(f"--- Loading: {file_path} ---")
        for key in data.files:
            val = data[key]
            shape_info = val.shape if hasattr(val, 'shape') else 'N/A'
            print(f"Key: {key:15} | Shape: {str(shape_info):15} | Type: {type(val)}")
        print("-" * 40)
    return data


def extract_field(data, field_name, verbose=False):
    """Extract NPZ field - keep as 2D (S, N) for core package compatibility."""
    if field_name in data:
        field = data[field_name]
        # Keep field as 2D (NumSlices, Nx*Ny) - core.reconstruct_3d_volume expects this format
        if field.ndim == 2:
            num_slices, values_per_field = field.shape
            n = int(np.sqrt(values_per_field))
            if n * n == values_per_field:
                if verbose:
                    print(f"Loaded {field_name}: {field.shape} (will reshape to ({num_slices}, {n}, {n}) internally)")
            else:
                if verbose:
                    print(f"Warning: Field {field_name} has {values_per_field} values, which is not a perfect square.")
        return field
    else:
        if verbose:
            print(f"Warning: Field '{field_name}' not found. Available keys: {list(data.keys())}")
        return None


def get_scalar(d, key, default):
    """Safely extract scalar value from npz dictionary."""
    if key in d:
        val = d[key]
        if val.size == 1:
            try:
                item = val.item()
                return float(item) if item is not None else default
            except:
                return float(val)
        return float(val[0])
    return default


def get_array(d, key):
    """Safely extract array from npz dictionary."""
    if key in d:
        val = d[key]
        if val.size == 1 and val.item() is None:
            return None
        return val
    return None


def resolve_parameters(args, data):
    """
    Resolve parameters from CLI args -> NPZ metadata -> Defaults.
    Returns resolved (total_height, iso_level, bounds_min, bounds_max).
    """
    # 1. Total Height
    file_height = get_scalar(data, "total_height", 10.0)
    total_height = args.height if args.height is not None else file_height
    
    # 2. Iso Level
    file_iso = get_scalar(data, "iso_level", 0.0)
    iso_level = args.iso if args.iso is not None else file_iso
    
    # 3. Bounds
    bounds_min = get_array(data, "bounds_min")
    bounds_max = get_array(data, "bounds_max")
    
    # Ensure 3D bounds with correct Z range based on total_height
    if bounds_min is not None:
        if len(bounds_min) == 2:
            bounds_min = np.array([bounds_min[0], bounds_min[1], 0.0])
        else:
            # Even if 3D, ensure Z_min is 0
            bounds_min = np.array([bounds_min[0], bounds_min[1], 0.0])
    
    if bounds_max is not None:
        if len(bounds_max) == 2:
            bounds_max = np.array([bounds_max[0], bounds_max[1], total_height])
        else:
            # Update Z_max to match total_height (don't trust NPZ Z bounds)
            bounds_max = np.array([bounds_max[0], bounds_max[1], total_height])
    
    return total_height, iso_level, bounds_min, bounds_max


def save_mesh_to_obj(verts, faces, output_path, verbose=False):
    """Save mesh to OBJ file with 1-indexed faces."""
    if verts is None or faces is None:
        if verbose:
            print("No mesh to save.")
        return False
        
    try:
        with open(output_path, "w") as f:
            f.write("# Narnia Exported Mesh (via core package)\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        if verbose:
            print(f"Mesh saved to: {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to write OBJ: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate mesh from Narnia SDF NPZ files using core package"
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to input .npz file (REQUIRED)")
    parser.add_argument("--field", type=str, default="profile_fields", help="Field name to extract")
    parser.add_argument("--iso", type=float, default=None, help="Iso-level (override)")
    parser.add_argument("--height", type=float, default=None, help="Total height (override)")
    parser.add_argument("--interp", type=int, default=2, help="Number of slice interpolations")
    parser.add_argument("--out", type=str, default=None, help="Output .obj path")
    parser.add_argument("--smooth", type=str, default=None, 
                       choices=['laplacian', 'taubin', 'combined'],
                       help="Smoothing method (requires Open3D)")
    parser.add_argument("--iterations", type=int, default=5, help="Smoothing iterations")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    
    args = parser.parse_args()
    
    # 1. Strict File Check
    if not os.path.exists(args.npz):
        sys.stderr.write(f"Error: Input file not found: {args.npz}\n")
        sys.exit(1)

    try:
        # 2. Load Data
        data = load_npz_file(args.npz, verbose=args.verbose)
        
        # 3. Resolve Parameters
        total_height, iso_level, bounds_min, bounds_max = resolve_parameters(args, data)
        
        # 4. Extract Field (keep as 2D)
        field_data = extract_field(data, args.field, verbose=args.verbose)
        if field_data is None:
            sys.stderr.write(f"Error: Field '{args.field}' not found in {args.npz}\n")
            sys.exit(1)
        
        # 5. Infer grid dimensions from 2D field shape
        num_slices, values_per_field = field_data.shape
        n = int(np.sqrt(values_per_field))
        if n * n != values_per_field:
            sys.stderr.write(f"Error: Field has {values_per_field} values, not a perfect square\n")
            sys.exit(1)
        nx, ny = n, n
        
        # 6. Reshape to 3D for interpolation (core.interpolate_slices expects 3D)
        field_data_3d = field_data.reshape(num_slices, ny, nx)
        
        # 7. Interpolate slices using core package
        if args.verbose:
            print(f"Interpolating slices (num_interpolations={args.interp})...")
        field_data_int = interpolate_slices(field_data_3d, num_interpolations=args.interp)
        
        # 8. Flatten back to 2D for reconstruct_3d_volume (expects 2D input)
        nz_int, ny_int, nx_int = field_data_int.shape
        field_data_2d = field_data_int.reshape(nz_int, ny_int * nx_int)
        
        if args.verbose:
            print(f"Reconstructing 3D volume: {field_data_2d.shape} -> ({nz_int}, {ny_int}, {nx_int}), height={total_height}")
        
        # 9. Reconstruct 3D volume using core package (expects 2D input, returns 3D volume)
        grid_data = reconstruct_3d_volume(
            field_data_2d,
            nx=nx_int,
            ny=ny_int,
            bounds_min=bounds_min,
            bounds_max=bounds_max
        )
        
        # 10. Generate Mesh using core package
        if args.verbose:
            print(f"Generating mesh for '{args.field}' at iso_level {iso_level}...")
        
        result = generate_mesh_marching_cubes(
            grid_data["volume"],
            grid_data["spacing"],
            grid_data["origin"],
            iso_level=iso_level
        )
        
        if result is None:
            sys.stderr.write("Error: Marching cubes generated no geometry.\n")
            sys.exit(1)
        
        verts, faces = result
        
        # 11. Optional Smoothing (NEW FEATURE)
        if args.smooth:
            if args.verbose:
                print(f"Applying {args.smooth} smoothing ({args.iterations} iterations)...")
            try:
                # Create temporary mesh dict for smoothing
                mesh_dict = {"vertices": verts, "triangles": faces}
                smoothed_dict = smooth_mesh(
                    mesh_dict,
                    method=args.smooth,
                    iterations=args.iterations
                )
                if smoothed_dict:
                    verts = smoothed_dict["vertices"]
                    faces = smoothed_dict["triangles"]
                else:
                    if args.verbose:
                        print("Warning: Smoothing failed, using original mesh")
            except ImportError:
                sys.stderr.write("Warning: Open3D not available, skipping smoothing\n")
            except Exception as e:
                sys.stderr.write(f"Warning: Smoothing failed: {e}\n")

        # 12. Save Output
        out_path = args.out if args.out else args.npz.replace(".npz", f"_{args.field}_core.obj")
        if save_mesh_to_obj(verts, faces, out_path, verbose=args.verbose):
            # Print ONLY the output path to stdout on success (GH capture)
            print(out_path)
            sys.exit(0)
        else:
            sys.stderr.write(f"Error: Failed to write output file: {out_path}\n")
            sys.exit(1)
            
    except Exception as e:
        sys.stderr.write(f"Error: Unexpected failure: {str(e)}\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # VSCode Debug Mode Check
    if len(sys.argv) == 1:
        print("--- DEBUG MODE DETECTED (No args provided) ---")
        
        debug_npz = "output\processed_sdf_results_20260113_173944.npz"
        
        if os.path.exists(debug_npz):
            sys.argv.append("--npz")
            sys.argv.append(debug_npz)
            sys.argv.append("--verbose")
            # Test smoothing feature
            # sys.argv.append("--smooth")
            # sys.argv.append("laplacian")
        else:
            import glob
            files = glob.glob("output/*.npz")
            if files:
                print(f"Debug target not found, using first available: {files[0]}")
                sys.argv.append("--npz")
                sys.argv.append(files[0])
                sys.argv.append("--verbose")
            else:
                print("Error: No debug target found. Please provide arguments or ensure .npz exists.")
                sys.exit(1)

    main()
