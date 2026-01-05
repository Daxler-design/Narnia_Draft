# Narnia SDF / Curve Viewer — Copilot Instructions

## Project overview
- This repo loads stacked scalar fields (SDF-like data), extracts iso-curves per slice, and visualizes them in an Open3D GUI viewer.
- Keep compute logic in `core.py` and UI logic in `narnia_vis.py`.
- Prefer minimal, surgical diffs that preserve current behavior unless the user explicitly asks for refactors.

## Key entry points
- Use `python main.py` as the primary entry point for local runs.
- Do not create an Open3D GUI context at import time; only initialize GUI inside `run_app*` flows.

## Repo map (for context)
- `main.py`: entry point; loads NPZ/JSON; launches viewer.
- `core.py`: scalar-field ops, boolean ops, iso-curves, centroid/voronoi utilities.
- `narnia_vis.py`: Open3D GUI app (tabs, callbacks, scene updates, export).
- `vis_utils.py`: stability env vars, monitor selection, Open3D helpers (LineSet, slice_z).
- `vis_widgets.py`: small GUI widget factories.

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
