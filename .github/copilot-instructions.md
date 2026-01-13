# Narnia SDF / Curve Viewer — Copilot Instructions

## Project overview
- This repo loads stacked scalar fields (SDF-like data), extracts iso-curves per slice, generates meshes via marching cubes, and visualizes them in an Open3D GUI viewer.
- **Refactored (Phase 1-6 complete)**: Core logic modularized into `core/` package, GUI helpers in `gui/` package.
- Keep compute logic in `core/` modules and UI logic in `narnia_vis.py`.
- Prefer minimal, surgical diffs that preserve current behavior unless the user explicitly asks for refactors.

## Key entry points
- Use `python main.py` as the primary entry point for local GUI runs.
- Use `MeshFromNPZ.py` (stable) or `examples/grasshopper_mesh_core.py` (experimental) for CLI mesh generation.
- Do not create an Open3D GUI context at import time; only initialize GUI inside `run_app*` flows.

## Repo map (for context)
- `main.py`: GUI entry point; loads NPZ/JSON; launches viewer.
- **core/** package (modular):
  - `data_utils.py`: NPZ/JSON loading, grid inference (3 functions)
  - `curves.py`: ISO-curve extraction (1 function)
  - `sdf_operations.py`: Boolean operations on SDFs (1 function)
  - `bracing_generator.py`: Voronoi-based bracing generation (6 functions)
  - `postprocess.py`: Morphological cleaning, temporal smoothing (4 functions)
  - `mesh_generator.py`: Marching cubes, smoothing, OBJ export (5 functions)
  - `__init__.py`: Public API (22 exported functions)
  - `README.md`: Module documentation with usage examples
- **gui/** package:
  - `widgets.py`: GUI widget factories
  - `mesh_builders.py`: Mesh building utilities for Open3D viewer
  - `__init__.py`: Package exports
- `narnia_vis.py`: Open3D GUI app (tabs, callbacks, scene updates, export).
- `vis_utils.py`: Stability env vars, monitor selection, Open3D helpers (LineSet, slice_z).
- **MeshFromNPZ.py**: STABLE production CLI tool for Grasshopper subprocess integration (DO NOT DELETE).
- **examples/**:
  - `grasshopper_mesh_core.py`: Enhanced CLI tool using core package (with smoothing feature).
  - `python_api_demo.py`: Library usage examples for scripts/notebooks.

## Grasshopper Integration
- `MeshFromNPZ.py` is the production-proven tool used in Grasshopper workflows via subprocess.
- `examples/grasshopper_mesh_core.py` demonstrates core package integration for testing new features (e.g., mesh smoothing).
- Both maintain identical CLI interface for drop-in compatibility.
- See `README.md` for complete Grasshopper subprocess pattern with caching and error handling.

## Stability and performance constraints
- Preserve the thread limiting approach (environment variables / threadpool limits) to avoid system freezes.
- Avoid heavy per-frame allocations in GUI callbacks; reuse arrays/objects where reasonable.
- Keep long-running computations off the GUI thread when possible; if UI must update, use Open3D’s main-thread posting pattern.

## Coding style
- Use type hints for public functions and non-trivial internal helpers.
- Add concise docstrings for new public functions, especially in `core.py`.
- Prefer `pathlib.Path` over raw string paths for filesystem work.
- Handle missing files and invalid inputs with clear error messages and non-crashing behavior.

## Open3D GUI conventions
- Keep all UI state inside the viewer class rather than module-level globals.
- When adding a new UI control, wire it with the existing patterns in `vis_widgets.py` and existing callbacks.

## Data and I/O conventions
- Keep relative paths relative to the repo root where possible.
- When exporting, prefer `.npz` with explicit keys and include bounds and iso level if available.
- Do not silently change data formats or key names; if changes are needed, provide a migration note.

## Validation checklist (when you propose changes)
- Explain how to run the app locally (`python main.py`) and what to manually verify in the UI.
- If you change `core.py`, suggest a small deterministic test case for the function(s).
- If you change file paths or exports, confirm backward compatibility with existing `.npz` files.

## Scope & non-goals
- Preserve behavior/UI flow unless explicitly requested.
- No drive-by refactors: no renaming sprees, no formatting-only diffs, no “cleanup” unrelated to the task.
- Keep to the existing 3-file structure; do not add new modules/files unless asked.

## Stepwise change protocol
- Always start with a short patch plan (bullets) before editing code.
- Implement the smallest change for that step (prefer extraction/moves over rewrites).
- End by listing: what changed + what intentionally did NOT change.

## Minimal smoke tests (must pass)
1) `python main.py` launches the window, aware virtual environment.
2) Compute tab: load/generate + slice/iso updates without crash.
3) Export `.npz` succeeds and preserves existing keys.
4) NPZ tab: load exported file + switch view + slice/iso updates without crash.
