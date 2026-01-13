# Narnia Core Package

Modular Python library for scalar field operations, bracing generation, and mesh generation from stacked 2D slices.

## Module Organization

The core package is split into focused modules for easier navigation and reuse:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **[data_utils.py](data_utils.py)** | Data loading and grid utilities | `stack_scalar_fields`, `meta_data_info`, `infer_grid_from_scalar_fields` |
| **[curves.py](curves.py)** | ISO-curve extraction for 2D visualization | `iso_curves_for_slice_2d` |
| **[sdf_operations.py](sdf_operations.py)** | Boolean operations on signed distance fields | `compute_sf_operation` |
| **[bracing_generator.py](bracing_generator.py)** | Voronoi-based structural bracing patterns | `generate_bracing_static`, `generate_bracing_keyfield_blend`, `compute_voronoi_sdf`, `generate_centroids`, `constrain_centroids_to_mask`, `get_profile_mask` |
| **[postprocess.py](postprocess.py)** | Morphological cleaning and temporal smoothing | `postprocess_bracing_fields`, `_binary_dilate`, `_binary_erode`, `_label_components` |
| **[mesh_generator.py](mesh_generator.py)** | Marching cubes mesh generation with Open3D | `interpolate_slices`, `reconstruct_3d_volume`, `generate_mesh_marching_cubes`, `smooth_mesh`, `export_mesh_obj` |

All public functions are exported through `core/__init__.py` for convenient imports:

```python
from core import (
    # Data loading
    stack_scalar_fields, meta_data_info, infer_grid_from_scalar_fields,
    
    # Mesh generation
    interpolate_slices, reconstruct_3d_volume, generate_mesh_marching_cubes,
    smooth_mesh, export_mesh_obj,
    
    # Bracing
    generate_bracing_static, generate_bracing_keyfield_blend,
    
    # Post-processing
    postprocess_bracing_fields,
    
    # Curves
    iso_curves_for_slice_2d,
    
    # SDF operations
    compute_sf_operation,
)
```

---

## Function Reference by Use Case

### 📁 Loading Data

**From NPZ files:**
```python
# Manual NPZ loading (or use examples/grasshopper_mesh_core.py)
data = np.load("output/results.npz")
field = data["profile_fields"]  # or "bracing_fields", "result_fields"
```

**From JSON files:**
```python
from core import stack_scalar_fields, meta_data_info

# Extract scalar field slices
scalar_fields_2d, keys = stack_scalar_fields(data_dict, prefix="scalar_field_values_")

# Extract metadata
iso_level, slice_count, bounds_max, bounds_min = meta_data_info(data_dict)
```

**Grid inference:**
```python
from core import infer_grid_from_scalar_fields

# Auto-detect nx, ny from array shape
num_fields, nx, ny = infer_grid_from_scalar_fields(scalar_fields_2d)
```

---

### 🏗️ Generating Bracing

**Static Voronoi bracing (fixed k centroids per slice):**
```python
from core import generate_bracing_static

bracing_fields = generate_bracing_static(
    profile_fields_2d,
    iso_level=0.0,
    nx=50,
    ny=50,
    k=5,              # Number of Voronoi cells
    seed=42           # Random seed for K-means
)
```

**Keyfield blending (variable centroids with smooth interpolation):**
```python
from core import generate_bracing_keyfield_blend

# Define key slices with varying centroid counts
keys_config = [
    (0, 3),            # Slice 0: 3 centroids
    (30, 5),           # Slice 30: 5 centroids
    (59, 4),           # Slice 59: 4 centroids
]

bracing_fields = generate_bracing_keyfield_blend(
    profile_fields_2d,
    iso_level=0.0,
    nx=50,
    ny=50,
    keys_config=keys_config,
    smooth=0.5,        # 50% smoothstep interpolation
    sigma=5.0,         # Ridge width parameter
    tau=0.5,           # Ridge threshold offset
    beta=4.0,          # Exponential blend softness
    seed=42
)
```

**Post-processing (morphological cleaning + temporal smoothing):**
```python
from core import postprocess_bracing_fields

cleaned_bracing = postprocess_bracing_fields(
    bracing_fields_2d,
    profile_fields_2d,
    iso_profile=0.0,
    iso_brace=0.0,
    close_radius=2,      # Morphological closing radius
    min_area=120,        # Remove islands < 120 pixels
    temporal_window=3    # Majority vote window (slices)
)
```

---

### 🔺 Generating Meshes

**Full pipeline: Interpolate → Volume → Marching Cubes → Smooth → Export:**
```python
from core import (interpolate_slices, reconstruct_3d_volume,
                  generate_mesh_marching_cubes, smooth_mesh, export_mesh_obj)

# 1. Interpolate slices for smoother mesh
interpolated = interpolate_slices(field_data, num_interpolations=2)

# 2. Reconstruct 3D volume
nz, ny, nx = interpolated.shape
bounds_min = np.array([0, 0, 0])
bounds_max = np.array([100, 100, 10])
grid_data = reconstruct_3d_volume(interpolated, nx, ny, bounds_min, bounds_max)

# 3. Generate mesh via marching cubes
result = generate_mesh_marching_cubes(
    grid_data["volume"],
    grid_data["spacing"],
    grid_data["origin"],
    iso_level=0.0
)

if result:
    vertices, faces = result
    
    # 4. Smooth mesh (optional)
    mesh_dict = {"vertices": vertices, "triangles": faces}
    smoothed = smooth_mesh(mesh_dict, method="taubin", iterations=5)
    
    # 5. Export to OBJ
    export_mesh_obj(smoothed, "output/mesh.obj")
```

---

### 📊 Extracting ISO-Curves (2D Visualization)

```python
from core import iso_curves_for_slice_2d
import numpy as np

# Get a single slice
slice_2d = field_data[30]  # Slice index 30
ny, nx = slice_2d.shape

# Create coordinate grids
x = np.linspace(0, 100, nx)
y = np.linspace(0, 100, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

# Extract curves at iso_level=0.0
curves = iso_curves_for_slice_2d(slice_2d, level=0.0, X=X, Y=Y)

# curves is a list of (N_i, 2) arrays [x, y]
for i, curve in enumerate(curves):
    print(f"Curve {i}: {len(curve)} points")
```

---

### 🔧 Boolean Operations on SDFs

```python
from core import compute_sf_operation

# Subtract bracing from profile
result = compute_sf_operation(
    profile_fields,
    bracing_fields,
    iso_level_A=0.0,
    iso_level_B=0.0,
    mode='difference'  # or 'union', 'intersection'
)

# Union (combine volumes)
combined = compute_sf_operation(sf_A, sf_B, mode='union')

# Intersection (overlapping region only)
overlap = compute_sf_operation(sf_A, sf_B, mode='intersection')

# Swap for "B minus A"
cavity = compute_sf_operation(sf_A, sf_B, mode='difference', swap=True)
```

---

## Workflow Patterns

### Pattern 1: NPZ → Mesh (Grasshopper Subprocess)

Use `examples/grasshopper_mesh_core.py` as a CLI tool from Grasshopper:

```python
# From Grasshopper Python component
import subprocess
result = subprocess.run([
    narnia_python_exe,
    "examples/grasshopper_mesh_core.py",
    "--npz", "data.npz",
    "--field", "profile_fields",
    "--iso", "0.0",
    "--smooth", "taubin",
    "--iterations", "5",
    "--out", "output.obj"
], capture_output=True, text=True)
print(result.stdout)  # Prints OBJ path on success
```

Or use stable production tool `MeshFromNPZ.py` (no smoothing feature).

### Pattern 2: JSON → Bracing → NPZ → Mesh (Python Script)

```python
from core import *
import json
import numpy as np

# 1. Load JSON
with open("data.json") as f:
    data_dict = json.load(f)

# 2. Extract fields
profile_fields_2d, _ = stack_scalar_fields(data_dict)
iso_level, _, _, _ = meta_data_info(data_dict)
num_fields, nx, ny = infer_grid_from_scalar_fields(profile_fields_2d)

# 3. Generate bracing
bracing = generate_bracing_static(profile_fields_2d, iso_level, nx, ny, k=5)
cleaned = postprocess_bracing_fields(bracing, profile_fields_2d, iso_level)

# 4. Save to NPZ
np.savez("output.npz", 
         profile_fields=profile_fields_2d,
         bracing_fields=cleaned,
         iso_level=iso_level)

# 5. Generate mesh (same as Pattern 1)
```

### Pattern 3: GUI Integration (Open3D Viewer)

See `narnia_vis.py` for full GUI implementation using:
- `iso_curves_for_slice_2d()` for 2D slice preview
- `generate_mesh_marching_cubes()` for 3D mesh preview
- Open3D SceneWidget for interactive visualization

---

## Dependencies

- **NumPy**: Array operations (all modules)
- **SciPy**: ndimage.zoom for interpolation (mesh_generator), spatial.cKDTree (bracing_generator)
- **scikit-image**: measure.marching_cubes (mesh_generator)
- **scikit-learn**: KMeans clustering (bracing_generator)
- **contourpy**: Fast contour generation (curves)
- **Open3D** *(optional)*: Mesh smoothing and export (mesh_generator)

---

## Design Principles

1. **Modular**: Each module has a focused purpose
2. **Type-safe**: Comprehensive type hints for all public functions
3. **Documented**: Detailed docstrings with Args, Returns, Examples, Notes
4. **Tested**: Refactored from proven production code (Phase 1-5)
5. **Backwards-compatible**: Public API preserved through `core/__init__.py`

---

## See Also

- **[../examples/python_api_demo.py](../examples/python_api_demo.py)**: Complete workflow examples
- **[../examples/grasshopper_mesh_core.py](../examples/grasshopper_mesh_core.py)**: CLI tool for Grasshopper integration
- **[../README.md](../README.md)**: Project overview and Grasshopper integration guide
- **[../MeshFromNPZ.py](../MeshFromNPZ.py)**: Stable production CLI tool (preserved)
