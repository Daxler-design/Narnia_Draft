# Phase 2 Plan: Local Backend API Skeleton

This phase focuses on the local-only backend that the WebUI will call. The goal is to
expose a stable API boundary over `core/` with clear request/response contracts.

## Objectives

1. **Create FastAPI service skeleton**
   - Local-only (`127.0.0.1`) and single-user.
   - Placeholder responses until compute integration is wired.

2. **Define request/response shapes**
   - Static bracing as the only active compute path for MVP.
   - Boolean + postprocess endpoints for parity.

3. **Document run instructions**
   - Simple local startup for designers.

## Deliverables

- `webui/backend/app.py` with MVP endpoints.
- `webui/backend/requirements.txt`.
- `webui/backend/README.md` with run instructions.

## Open Questions

1. Should the backend store intermediate results on disk (cache folder), or keep
   everything in memory for MVP?
2. For local uploads, where do you want cached files stored (e.g., `./output/webui_cache`)?

## Next Steps After Approval

- Integrate `core/` compute functions with the static bracing endpoint.
- Add preview payload formats for slice/curve/mesh data.
