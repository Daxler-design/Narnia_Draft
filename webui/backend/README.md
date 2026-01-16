# Narnia WebUI Backend (Phase 2 Skeleton)

This backend is a local-only FastAPI skeleton intended to support the WebUI MVP.
Static bracing compute is wired to `core/`, while preview/mesh/export remain placeholders.
The backend loads core modules directly to avoid requiring Open3D on the server.

## Run (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Notes

- Endpoints are intentionally minimal placeholders and will be connected to `core/`.
- This service is designed for single-user local workflows.
- Cache output is written to `./output/webui_cache`.
