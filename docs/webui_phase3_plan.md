# Phase 3 Plan: WebUI MVP (Frontend)

This phase delivers the WebUI with parity to the current desktop GUI while keeping
only static bracing compute active.

## Objectives

1. **UI parity with desktop GUI**
   - Same tabs (Compute / NPZ Viewer / Mesh).
   - All controls present, non-static compute panels disabled.

2. **Viewport parity**
   - Mesh + curves + field slice previews aligned with the existing GUI.

3. **Local-only workflow**
   - File uploads store into a local cache folder.
   - Optional path input fields for power users.

## Proposed Stack

- **React + Vite** for UI
- **Three.js** for rendering
- **WebGPU** (optional) via Three.js renderer, with WebGL fallback

## Deliverables

- `webui/frontend/` scaffold (Vite project)
- UI panels and controls mapped 1:1 to Phase 1 checklist
- Basic preview rendering with placeholder data

## Decisions

1. **Packaging**: browser-only (no Electron shell).

## Next Steps After Approval

- Scaffold the frontend project and wire it to the Phase 2 backend.
