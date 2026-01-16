# WebUI MVP Plan (Phase 1–3)

This document captures the current Open3D GUI controls (as the parity target) and a proposed
WebUI MVP plan using a local-only backend + frontend split. The MVP focuses on full UI parity
with the existing GUI while initially enabling **only the static bracing** compute path.

Sources of truth:
- `narnia_vis.py` for current GUI controls and layout.
- `README.md` for overall architecture and pipeline overview.

## 1) Current GUI Controls (Parity Checklist)

### Tab: **Compute**

**Input paths**
- Profile JSON path (file picker)
- Bracing JSON path (file picker; disabled when "Generate Bracing" is enabled)

**Bracing generation**
- Toggle: Generate Bracing (on/off)
- Generation Method (combobox)
  - static-bracing
  - Key-Field Blend
  - Shape-Adaptive
  - Binary Splitting

**Static Bracing Params**
- Num Centroids (slider, int)
- Ridge Width (Sigma) (slider)

**Key-Field Blend Params**
- Keys (Text input: "slice:k, ...")
- Smooth Factor (slider)
- Sigma / Ridge Width (slider)
- Tau / Threshold (slider)
- Beta / Blend Softness (slider)

**Shape-Adaptive Params**
- Area per Seed (px²) (slider)
- K Min (slider, int)
- K Max (slider, int)
- Z Smooth (sigma) (slider)
- Ramp Slices (slider, int)

**Binary Splitting Params**
- K Start (cells) (combo: 1,2,4,8,16)
- K Max (cells) (combo: 1,2,4,8,16,32)
- Split Offset (slider)
- Z Smooth (sigma) (slider)
- Ramp Slices (slider, int)

**Compute action**
- Button: Load / Re-generate Bracing

**Postprocess Bracing**
- Enable Postprocess (checkbox)
- Close Radius (slider)
- Min Area (slider)
- Temporal Window (slider, int)

**Boolean Operation**
- Mode (combobox: difference / union / intersection)
- Profile Offset (slider)
- Bracing Offset (slider)
- Result Iso threshold (slider)

---

### Tab: **NPZ Viewer**

**Input**
- NPZ File Path (file picker)
- Button: Load NPZ File

**Viewer settings**
- View Channel (combobox: Result)
- Result Iso threshold (slider)

Note: "Loading NPZ skips compute."

---

### Tab: **Mesh**

**Source**
- Source (combobox: Compute / NPZ Viewer / Custom)
- Button: Load Custom NPZ...

**Include**
- Result Mesh (checkbox)
- Profile Mesh (checkbox)
- Bracing Mesh (checkbox)

**Marching Cubes Parameters**
- Total Height (slider)
- Z Interpolation (slider, int)
- Slice count info label

**Iso Overrides**
- Result Iso Override (slider)
- Profile Iso Override (slider)
- Bracing Iso Override (slider)

**Smoothing**
- Method (combobox: None / Laplacian / Taubin / Laplacian + Taubin)
- Iterations (slider, int)

**Actions**
- Fit Camera (button)
- Update Mesh (button)

**Export**
- Export Location (file picker)
- Export Mesh (.obj) (button)

---

### Shared (All Tabs; hidden in Mesh tab)

**Visualization**
- Slice (int slider)
- Fit Camera (button)

**Export**
- Output Directory (text input + folder selector)
- Export Profile (checkbox)
- Export Bracing (checkbox)
- Export Results (.npz) (button)

**Status**
- Status label

## 2) WebUI MVP Plan (Phase 1–3)

### Phase 1: Parity Spec + UI Wireframe (1–2 days)

**Goal:** translate the GUI control list into a single WebUI parity spec.

**Deliverables**
- This document as baseline control list.
- Low-fidelity wireframe mapping each control to a WebUI panel.
- A control-to-API mapping (inputs → backend requests → outputs).

### Phase 2: Local Backend API (2–4 days)

**Goal:** define a local-only API boundary over `core/` (future-proof for C++).

**Proposed backend framework:** **FastAPI**

**Key endpoints (MVP):**
- `POST /compute/static-bracing` — run compute with static bracing parameters.
- `POST /compute/boolean` — boolean operation + offsets.
- `POST /compute/postprocess` — optional postprocess on bracing fields.
- `GET /result/preview` — slice + iso preview (curves/scalar view).
- `POST /mesh/generate` — marching cubes mesh generation.
- `GET /mesh/{id}` — fetch mesh asset (OBJ/PLY).
- `POST /export/npz` — export results to NPZ.

**Data contracts (MVP):**
- Mesh: `.obj` or `.ply`
- Scalars/fields: `.npz`
- Metadata: JSON (bounds, iso levels, slice count)

**Local-only constraints:**
- Bind to `127.0.0.1` (no multi-user auth).
- Local file paths are allowed (trusted environment).

### Phase 3: WebUI MVP (1–2 weeks)

**Goal:** implement UI parity with current GUI, but only enable static bracing compute.

**Proposed frontend framework:**
- **Vite + React** for UI
- **Three.js** for mesh display + line rendering
- **WebGPU** optional via Three.js WebGPU renderer (feature flag), fallback to WebGL.

**UI parity scope:**
- All panels and controls present, **but only static bracing** active in compute.
- Non-static panels shown but disabled or marked "Coming Soon".

**Visualization scope:**
- Mesh preview: triangle mesh render
- Curve preview: line rendering for contours/iso-curves
- Scalar field preview: slice-based texture rendering (2D slice in 3D plane)

**State flow:**
- UI collects parameters → calls API → stores results in local cache → renders.

## 3) Deployment Notes (Local Only)

**Local run workflow (MVP):**
- `uvicorn app:app --host 127.0.0.1 --port 8000`
- `npm run dev` for WebUI (or `npm run build` + static hosting)

**No multi-user or auth** is required for MVP (single-user local usage).

## 4) MVP Acceptance Criteria

- All GUI panels and controls exist in the WebUI.
- Static bracing compute path is functional end-to-end.
- Mesh/curve/field previews are visible and interactive.
- Exports match current desktop output formats.

