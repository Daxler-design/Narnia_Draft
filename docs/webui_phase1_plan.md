# Phase 1 Plan: WebUI Parity Spec + Wireframe

This plan focuses on Phase 1 (parity spec + wireframe) for the WebUI MVP. The goal is to
lock down a shared, reviewable specification before we build backend or frontend code.

## Objectives

1. **Parity checklist finalized**
   - Confirm every Open3D GUI panel/control that must exist in the WebUI.
   - Identify which controls are active for MVP (static bracing) vs. disabled (other methods).

2. **WebUI screen/flow definition**
   - Define screen layout and navigation (tabs vs. side panels).
   - Map existing GUI tab structure (Compute / NPZ Viewer / Mesh) to WebUI routes or tabs.

3. **Control-to-API mapping**
   - Document which user actions trigger backend requests.
   - Specify required request/response payload fields (no implementation yet).

4. **Wireframe draft**
   - Provide a simple layout diagram (text-based) for review.
   - Validate that all existing controls have a WebUI home.

## Deliverables

- Updated parity checklist if needed (new changes appended to `docs/webui_mvp_plan.md`).
- This Phase 1 plan document, plus a reviewed wireframe layout.
- A control-to-API mapping table for static bracing path.

## Proposed WebUI Layout (Wireframe Draft)

```
+---------------------------------------------------------------+
| Header: Narnia WebUI (Local)                                  |
+--------------------+------------------------------------------+
| Left Sidebar       | Main Viewport (3D)                       |
| - Tabs:            | - Mesh / Curves / Field slice            |
|   [Compute]        | - Orbit / pan / zoom                     |
|   [NPZ Viewer]     | - Status overlay                         |
|   [Mesh]           |                                          |
|                    |                                          |
| Panel Area         |                                          |
| - Control groups   |                                          |
| - Buttons/Sliders  |                                          |
+--------------------+------------------------------------------+
| Footer: status + export hints                                 |
+---------------------------------------------------------------+
```

## Control-to-API Mapping (Draft)

| UI Action | Backend Endpoint | Notes |
|----------|------------------|-------|
| Load / Re-generate Bracing | `POST /compute/static-bracing` | Only static method active for MVP |
| Toggle Generate Bracing | Client-side | Toggles path input + options |
| Boolean Operation change | `POST /compute/boolean` | Requires profile+bracing data loaded |
| Postprocess toggle/params | `POST /compute/postprocess` | Applies to bracing fields |
| Slice change | `GET /result/preview` | Returns slice preview data |
| Update Mesh | `POST /mesh/generate` | Marching cubes + smoothing |
| Export NPZ | `POST /export/npz` | Saves to output dir |
| Export Mesh | `POST /mesh/export` | Saves to export location |

## Open Questions (Need Your Input)

1. **Wireframe format**: Do you want a simple text wireframe (as above), or a diagram (PNG)?
2. **Tab behavior**: Should WebUI preserve the same tab structure (Compute / NPZ Viewer / Mesh),
   or can we merge panels into a single view with collapsible sections?
3. **Preview composition**: For MVP, should the viewport show **mesh + curves + slice**
   simultaneously, or should we have a selector to toggle modes?
4. **File access**: Is it acceptable to keep local file-path inputs (like the desktop GUI),
   or should the WebUI add file upload/select dialogs?

## Next Steps After Approval

- Update `docs/webui_mvp_plan.md` if you approve changes to parity or layout.
- Start Phase 2: define backend API skeleton + local runtime.

