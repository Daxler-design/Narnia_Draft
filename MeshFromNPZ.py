"""
MeshFromNPZ.py was refactored into a Grasshopper-friendly CLI tool 
that converts stacked 2D scalar fields stored in a Narnia NPZ into a triangulated OBJ mesh. 

MeshFromNPZ

# Key logic flow:

Load the NPZ (optionally verbose, printing keys/shapes only).

Resolve parameters with a clear priority: CLI overrides → NPZ metadata (iso_level, total_height, bounds) → hard defaults.

Extract a chosen field (e.g., profile_fields, bracing_fields) and reshape flattened slices from (S, N) into (S, n, n) when N is a perfect square.

Optionally densify along Z via linear interpolation between slices to improve marching-cubes continuity.

Reconstruct a 3D volume with explicit Z spacing from total_height and XY spacing from bounds when available.

Run skimage.measure.marching_cubes and convert vertex order from (z,y,x) to (x,y,z), applying origin offset.

Save an OBJ (1-indexed faces) and print only the output path on success; errors go to stderr with nonzero exit codes.

# Purpose:

Keep heavy dependencies (numpy/scipy/skimage) outside Rhino/GH,
enable deterministic command-line execution from a subprocess, and make output capture + caching straightforward on the Grasshopper side.
"""




import numpy as np
import scipy.ndimage as ndimage
from skimage import measure
import os
import argparse
import sys

# load and read npz, parse and print npz structure, understand what the data it would contain
def load_npz_file(file_path, verbose=False):
    # load the npz file
    data = np.load(file_path, allow_pickle=True)
    if verbose:
        print(f"--- Loading: {file_path} ---")
        for key in data.files:
            val = data[key]
            shape_info = val.shape if hasattr(val, 'shape') else 'N/A'
            print(f"Key: {key:15} | Shape: {str(shape_info):15} | Type: {type(val)}")
        print("-" * 40)
    return data





# reconstruct the grid from planar slices,since the npz contains 2d slices of sdf values
# there is no z info in the file, so we need define the z axis and value sepereatly
def reconstruct_3d_grid_from_slices(field_data, total_height, bounds_min=None, bounds_max=None):
    if field_data is None:
        return None
        
    nz, ny, nx = field_data.shape
    
    # Calculate spacing
    dz = total_height / (nz - 1) if nz > 1 else 1.0
    
    dx = 1.0
    dy = 1.0
    origin = [0, 0, 0]
    
    if bounds_min is not None and bounds_max is not None:
        # Assuming nx is X, ny is Y
        dx = (bounds_max[0] - bounds_min[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (bounds_max[1] - bounds_min[1]) / (ny - 1) if ny > 1 else 1.0
        origin = [bounds_min[0], bounds_min[1], 0]

    return {
        "volume": field_data,
        "spacing": (dz, dy, dx), # Z, Y, X spacing
        "origin": origin
    }

# there are different fields in the npz file, we need choose one of them to generate the mesh
# `result_fields`, `profile_fields`, `bracing_fields`
def extract_field(data, field_name, verbose=False):
    if field_name in data:
        field = data[field_name]
        # Many fields are saved as flattened (NumSlices, Nx*Ny)
        if field.ndim == 2:
            num_slices, values_per_field = field.shape
            n = int(np.sqrt(values_per_field))
            if n * n == values_per_field:
                if verbose:
                    print(f"Reshaping {field_name} from {field.shape} to ({num_slices}, {n}, {n})")
                field = field.reshape(num_slices, n, n)
            else:
                if verbose:
                    print(f"Warning: Field {field_name} has {values_per_field} values, which is not a perfect square.")
        return field
    else:
        if verbose:
            print(f"Warning: Field '{field_name}' not found. Available keys: {list(data.keys())}")
        return None



# handle sice-to-slice interpolation for continuous mesh generation
# between slice i and slice i+1, there should be interpolation to provide enough data for marching cubes
def interpolate_slices(field_data, num_interpolations, verbose=False):
    if field_data is None:
        return None
    
    if num_interpolations <= 0:
        return field_data
        
    # field_data shape is (Z, H, W)
    # We increase resolution in Z (axis 0)
    nz, ny, nx = field_data.shape
    new_nz = nz + (nz - 1) * num_interpolations
    
    # Calculate zoom scale for Z axis
    zoom_scale = new_nz / nz
    
    # Linear interpolation (order=1) along Z axis only
    if verbose:
        print(f"Interpolating: {nz} slices -> {new_nz} slices...")
    interpolated = ndimage.zoom(field_data, (zoom_scale, 1, 1), order=1)
    
    return interpolated


# generate mesh using marching cubes from the reconstructed 3d grid
def generate_mesh_from_grid(grid_data, iso_level=0.0):
    if grid_data is None:
        return None, None
        
    volume = grid_data["volume"]
    spacing = grid_data["spacing"]
    origin = grid_data["origin"]
    
    # skimage marching cubes
    # It returns vertices (Z, Y, X) order if volume is (Z, Y, X)
    try:
        verts, faces, normals, values = measure.marching_cubes(volume, level=iso_level, spacing=spacing)
    except ValueError as e:
        print(f"Error in Marching Cubes: {e}")
        v_min, v_max = np.min(volume), np.max(volume)
        print(f"Volume range: [{v_min:.4f}, {v_max:.4f}]. Target level: {iso_level}")
        return None, None
    
    # Vertices returned are (z, y, x) in spatial coordinates scaled by spacing.
    # Reorder to (x, y, z) and add origin
    verts_xyz = np.zeros_like(verts)
    verts_xyz[:, 0] = verts[:, 2] + origin[0] # X
    verts_xyz[:, 1] = verts[:, 1] + origin[1] # Y
    verts_xyz[:, 2] = verts[:, 0] + origin[2] # Z
    
    return verts_xyz, faces


def save_mesh_to_obj(verts, faces, output_path, verbose=False):
    if verts is None or faces is None:
        if verbose: print("No mesh to save.")
        return False
        
    try:
        with open(output_path, "w") as f:
            f.write("# Narnia Exported Mesh\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        if verbose:
            print(f"Mesh saved to: {output_path}")
        return True
    except Exception as e:
        if verbose: print(f"Failed to write OBJ: {e}")
        return False






def get_scalar(d, key, default):
    """Safely extract scalar value from npz dictionary"""
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
    """Safely extract array from npz dictionary"""
    if key in d:
        val = d[key]
        if val.size == 1 and val.item() is None:
            return None
        return val
    return None

def resolve_parameters(args, data):
    """
    Resolve parameters from CLI args -> NPZ metadata -> Defaults
    Returns resolved (total_height, iso_level, bounds_min, bounds_max)
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
    
    return total_height, iso_level, bounds_min, bounds_max

def main():
    parser = argparse.ArgumentParser(description="Generate mesh from Narnia SDF NPZ files")
    parser.add_argument("--npz", type=str, required=True, help="Path to input .npz file (REQUIRED)")
    parser.add_argument("--field", type=str, default="profile_fields", help="Field name to extract")
    parser.add_argument("--iso", type=float, default=None, help="Iso-level (override)")
    parser.add_argument("--height", type=float, default=None, help="Total height (override)")
    parser.add_argument("--interp", type=int, default=2, help="Number of slice interpolations")
    parser.add_argument("--out", type=str, default=None, help="Output .obj path")
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
        
        # 4. Extract & Process Field
        field_data = extract_field(data, args.field, verbose=args.verbose)
        if field_data is None:
            sys.stderr.write(f"Error: Field '{args.field}' not found in {args.npz}\n")
            sys.exit(1)
            
        field_data_int = interpolate_slices(field_data, num_interpolations=args.interp, verbose=args.verbose)
        
        # 5. Reconstruct Grid
        grid_info = reconstruct_3d_grid_from_slices(
            field_data_int, 
            total_height=total_height, 
            bounds_min=bounds_min, 
            bounds_max=bounds_max
        )
        
        # 6. Generate Mesh
        if args.verbose:
            print(f"Generating mesh for '{args.field}' at iso_level {iso_level}...")
            
        verts, faces = generate_mesh_from_grid(grid_info, iso_level=iso_level)
        
        if verts is None:
            sys.stderr.write("Error: Marching cubes generated no geometry.\n")
            sys.exit(1)

        # 7. Save Output
        out_path = args.out if args.out else args.npz.replace(".npz", f"_{args.field}.obj")
        if save_mesh_to_obj(verts, faces, out_path, verbose=args.verbose):
            # Print ONLY the output path to stdout on success (easy for GH to capture)
            print(out_path)
            sys.exit(0)
        else:
            sys.stderr.write(f"Error: Failed to write output file: {out_path}\n")
            sys.exit(1)
            
    except Exception as e:
        sys.stderr.write(f"Error: Unexpected failure: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    # VSCode Debug Mode Check
    # If no arguments are passed (just script name), assume debug mode and inject default arguments.
    # This allows F5 debugging in VSCode without messing up the CLI for Grasshopper.
    if len(sys.argv) == 1:
        print("--- DEBUG MODE DETECTED (No args provided) ---")
        
        # Hardcoded debug values - change these to test specific files/cases
        debug_npz = "output/processed_sdf_results_20260112_130342.npz" 
        
        if os.path.exists(debug_npz):
            sys.argv.append("--npz")
            sys.argv.append(debug_npz)
            sys.argv.append("--verbose")
            # You can add other debug args here
            # sys.argv.append("--iso")
            # sys.argv.append("0.1")
        else:
             # Try to find *any* npz in output/ for convenience
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
