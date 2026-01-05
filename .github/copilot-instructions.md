# Narnia – Cell-Wall Bracing (Hybrid OT + Field) Implementation Guide

This guide is a step-by-step plan to replace **per-slice Voronoi-ridge bracing** with **continuous 3D “cell walls”** that:
- follow the cavity/profile geometry (no “emerge out of nowhere”),
- transition smoothly across slices,
- support varying cell count/density along Z,
- produce a mesh-ready implicit field.

The approach is **field-first**:
> Build a *single* coherent bracing **volume** (implicit field) and slice it, rather than building unrelated 2D drawings.

---

## 0. Vocabulary

- **Profile field** `P[z,y,x]`: your existing scalar field stack (SDF-ish).
- **Inside mask** `M[z,y,x]`: `P < iso_level`.
- **Seeds / generators** `S[z,k,2]`: cell centers per slice with stable IDs.
- **Soft assignment** `Q[z,k,y,x]`: probability (or weight) of region k at each pixel.
- **Wall field** `W[z,y,x]`: high values where multiple regions compete (cell boundaries).
- **Wall SDF** `Bw[z,y,x]`: signed distance-like field of walls (after thickening).
- **Final solid** `F`: combined (double wall + internal cell walls) implicit volume.

---

## 1. Key Design Choices (to prevent “popping”)

### 1.1 Use *soft* regions, not hard Voronoi labels
Hard Voronoi diagrams can change topology abruptly when seeds move.  
Instead, compute **softmax over distances**:

- Distance to seed k: `Dk(x) = ||x - s_k||`
- Soft region weight: `Qk(x) = softmax(-Dk / tau)`

`tau` controls sharpness.

### 1.2 Define walls from **competition**, not from a particular ridge formula
Cell walls appear where at least two regions have similar weights.

Good wall strength definitions:
- `W = 1 - (top1 - top2)`  (needs normalization)
- `W = entropy(Q)`  (high where uncertain)
- `W = 1 / (eps + (d2 - d1))` from weighted distances (optional)

### 1.3 Make the process 3D-consistent
After you compute `W[z,y,x]`, apply **z-regularization**:
- `gaussian_filter(W, sigma=(sigma_z, sigma_y, sigma_x))` with small `sigma_z`
- or 1D smoothing along Z only.

---

## 2. Implementation Milestones (recommended order)

### Milestone A — Cell wall field per slice (no OT yet)
Goal: replace `compute_voronoi_sdf()` usage with a **cell-wall field** that looks meaningful in one slice.

Deliverables:
- `compute_soft_regions(XY, seeds, tau)` → per-pixel soft weights `Qk`
- `compute_wall_strength(Qk)` → `W`
- `thicken_wall_field(W, thickness_px)` → bracing field `B`

Acceptance:
- On a single slice, walls align between cells and are not speckled.
- `B` can be combined with your existing boolean ops.

### Milestone B — Stable seeds across slices (no OT yet)
Goal: remove slice-to-slice random changes without introducing weird births.

Deliverables:
- stable seed IDs via Hungarian assignment between consecutive slices,
- a conservative “split” rule to add seeds where needed,
- a schedule for target cell counts per slice (K schedule).

Acceptance:
- seeds move smoothly across z,
- adding a seed occurs in a predictable place (largest cell / farthest-point within a region),
- walls don’t jump.

### Milestone C — OT-guided transport (seed advection with geometry)
Goal: seeds follow geometry deformation instead of “sliding” independently.

Deliverables:
- `compute_ot_map(mask_i, mask_{i+1})` producing sparse correspondences,
- `fit_dense_warp(corr)` producing a smooth displacement field `u(x)`,
- apply warp to advect seeds: `s_{i+1,pred} = s_i + u(s_i)`.

Acceptance:
- if the cavity shifts/rotates, walls follow it with minimal lag,
- far fewer “unexplained” changes.

### Milestone D — 3D wall volume + final meshing hook
Goal: produce a watertight 3D implicit bracing volume ready for meshing.

Deliverables:
- `W[z,y,x]` full volume,
- `Bw[z,y,x]` (thickened wall solid field),
- (optional) marching cubes export.

Acceptance:
- bracing looks continuous when scrubbing slices,
- exported mesh is continuous (no holes from inconsistent slices).

---

## 3. Parameter Recommendations (starting points)

- `tau` (softmax temperature): 2–6 pixels
- `sigma_z` (z smoothing): 0.6–1.2 slices
- `thickness_px` (wall thickness): 1–3 pixels (later convert to mm)
- K schedule: linear or piecewise (slow growth early, faster mid)

---

## 4. OT Module (practical version)

OT is used for **transport**, not for “generating walls”.

**Minimal OT**:
1. Sample N points from inside mask on slice i and i+1 (or boundary band).
2. Compute cost matrix `C = ||xi - yj||^2`.
3. Solve entropic Sinkhorn for coupling `Pi`.
4. Compute barycentric map for each xi: `T(xi) = Σ_j Pi_ij * yj / Σ_j Pi_ij`.
5. Fit a smooth warp from correspondences (thin-plate / RBF) to a dense displacement field.

Use SciPy:
- `scipy.interpolate.Rbf` (thin-plate) or a small custom TPS solve.

---

## 5. Integration Notes (keep viewer stable)

- Keep `generate_bracing_json(...)` but allow a new mode: `"cell_walls"`.
- Keep the output as scalar fields per slice so the viewer (contours) keeps working.
- Do *not* attempt to interpolate contours directly. Always build fields first.

---

## 6. Debug Outputs (high value)
Export these arrays to NPZ:
- `seeds[z,k,2]`
- `Q` summary: top1/top2 per slice
- `W[z]` wall strength
- adjacency stability metrics

---

## 7. Common Failure Modes & Fixes

- **Walls look noisy** → increase `tau`, add XY smoothing, enforce seeds inside mask band.
- **Walls fade in/out** → increase z-smoothing, use transport (OT), avoid per-slice re-init.
- **New walls pop** → births must be farthest-point inside largest cell + ramp tau locally.
- **Walls cross boundary** → clamp `Q` to 0 outside mask, renormalize.

---

## 8. What “Done” Looks Like
- Scrubbing slices shows walls that bend/shift, not teleport.
- New walls appear as a gradual split of a larger cell, not from empty space.
- A 3D mesh extraction produces a continuous internal cell-wall network between double walls.

