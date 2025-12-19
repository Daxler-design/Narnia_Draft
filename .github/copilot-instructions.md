# SDF Bracing Generation Codebase Guide

## Project Overview

This is a **Signed Distance Field (SDF) post-processing pipeline** for architectural brace/structural bracing design. The workflow processes volumetric SDF data stored in JSON format, performs geometric operations, generates Voronoi-based bracing patterns, and exports results for downstream use.

## Setup
- Install runtime dependencies with `pip install -r requirements.txt` before running the scripts or viewer from another machine.

## Architecture & Data Flow

```
JSON Input (SDF Stack Fields)
    ↓
[core.py] Parse & Stack → 2D scalar arrays (num_fields × values_per_field)
    ↓
[main.py] Load fields → validate shape/bounds
    ↓
[core.py] Boolean ops (union/intersection/difference)
    ↓
[core.py] Generate Voronoi bracing (K-Means centroids → Voronoi SDF)
    ↓
[core.py] Extract Iso-curves (matplotlib contour engine)
    ↓
Output: NPZ file with result fields, JSON/NPZ with curves
```

## Key Components & Patterns

### 1. **core.py** - Integrated Logic
Consolidates all geometric and data processing logic (formerly in `sf_tools.py` and `sf_bracing.py`).
- **Data Handling**: `stack_scalar_fields`, `meta_data_info`, `infer_grid_from_scalar_fields`.
- **Geometric Ops**: `compute_sf_operation` (Boolean SDF), `iso_curves_for_slice_2d`.
- **Bracing Logic**: `get_profile_mask`, `generate_centroids` (K-Means), `compute_voronoi_sdf`, `constrain_centroids_to_mask`.

### 2. **main.py** - Production Pipeline
The primary entry point for batch processing.
- Orchestrates the full workflow: Load → Stack → Boolean Op → Curve Extraction → Save.
- Designed for CLI/headless execution (no GUI/plotting).
- Saves results to `.npz` for downstream CAD/analysis.

### 3. **notebook_preview.py** - Visualization Utilities
Contains all functions intended for interactive use in Jupyter Notebooks.
- **`plot_scalar_overview()`**: Histogram + thumbnail grid for field inspection.
- **`show_slice()`**: Detailed view of a single slice with contour overlays.

### 4. **251121_sf_postProcess.ipynb** - Interactive Exploration
- Used for prototyping and parameter tweaking.
- Imports from `core.py` and `notebook_preview.py`.

## Data Format Conventions

### JSON Input Structure
```json
{
  "iso_level": 0.0,
  "slice_count": 50,
  "bounds_min": [0, 0, 0],
  "bounds_max": [100, 100, 100],
  "scalar_field_values_0": [...],
  ...
}
```

### Grid Assumption
- Scalar fields flatten to 1D: `values_per_field = nx × ny`.
- Grid is **always square**: `nx == ny == sqrt(values_per_field)`.
- Grid indexing: `(ny, nx)` = `(rows, cols)` = array indexing; centroid format `[y, x]`.

## Common Tasks & Patterns

### Adding a New Field Operation
1. Implement in `core.py` following `compute_sf_operation()` signature.
2. Call from `main.py` or notebook.

### Debugging Shape Mismatches
- Always call `core.infer_grid_from_scalar_fields()` first to validate square grid.
- Check JSON: ensure `slice_count × sqrt(values_per_field)` matches expectations.

## Performance Notes

- **Windows OpenMP issue**: Set `os.environ["OMP_NUM_THREADS"] = "1"` if using `sklearn` in parallel contexts.
- **K-Means coherence**: Use `prev_centroids` in `generate_centroids` to prevent jumping between slices.
- **Iso-curve extraction**: Uses `matplotlib.pyplot` internally; `core.py` handles figure cleanup to prevent memory leaks.
