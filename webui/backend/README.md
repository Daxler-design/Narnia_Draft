# Narnia WebUI Backend (Phase 2 Skeleton)

This backend is a local-only FastAPI skeleton intended to support the WebUI MVP.
Static bracing, boolean, and postprocess compute are wired to `core/`, while preview/mesh/export remain placeholders.
The backend loads core modules directly to avoid requiring Open3D on the server.

## Run (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Notes

- Endpoints are intentionally minimal; preview/mesh/export are still placeholders.
- This service is designed for single-user local workflows.
- Cache output is written to `./output/webui_cache`.

## Smoke Tests (PowerShell)

```powershell
$base = "http://127.0.0.1:8000"
$profile = "D:/OneDrive/Work/ZahaWork/Research/Narnia/Narnia_Draft/alice_result/251120/ext/waveStackFields.json"

$staticPayload = @{
  profile_json_path = $profile
  generate_bracing = $true
  params = @{ num_centroids = 5; ridge_sigma = 0.0 }
} | ConvertTo-Json -Depth 4

$staticResp = curl.exe -s -X POST "$base/compute/static-bracing" `
  -H "Content-Type: application/json" -d $staticPayload | ConvertFrom-Json
$datasetId = $staticResp.dataset_id

$postprocessPayload = @{
  enable = $true
  close_radius = 2.0
  min_area = 120.0
  temporal_window = 3
  profile_offset = 0.0
  bracing_offset = 0.0
  dataset_id = $datasetId
} | ConvertTo-Json -Depth 4
curl.exe -s -X POST "$base/compute/postprocess" `
  -H "Content-Type: application/json" -d $postprocessPayload

$booleanPayload = @{
  mode = "difference"
  profile_offset = 0.0
  bracing_offset = 0.0
  result_iso = 0.0
  dataset_id = $datasetId
} | ConvertTo-Json -Depth 4
curl.exe -s -X POST "$base/compute/boolean" `
  -H "Content-Type: application/json" -d $booleanPayload
```

Update `profile` to point at a valid `waveStackFields.json` on your machine.

### Preview Smoke Call (PowerShell)

```powershell
$preview = Invoke-RestMethod -Method Get -Uri "$base/result/preview?dataset_id=$datasetId&slice_index=0&iso_threshold=0.0"
$preview | Select-Object dataset_id, slice_index, slice_count, z_coord, iso_threshold
$preview.fields.result.Count, $preview.fields.result[0].Count
$preview.curves.result.Count, $preview.curves.profile.Count, $preview.curves.bracing.Count
```

Expected payload shape:
- `fields.result`, `fields.profile`, `fields.bracing` are 2D arrays (ny x nx).
- `curves.*` are lists of polylines; each polyline is an array of `[x, y]` points.
