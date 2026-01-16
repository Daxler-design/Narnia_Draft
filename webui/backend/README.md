# Narnia WebUI Backend (Phase 2 Skeleton)

This backend is a local-only FastAPI skeleton intended to support the WebUI MVP.
Endpoints currently return placeholder payloads while the compute integration is wired
up in later phases.

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
