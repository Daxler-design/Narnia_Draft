from __future__ import annotations

import json
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "core"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bracing_generator = _load_module("bracing_generator", CORE_DIR / "bracing_generator.py")
_data_utils = _load_module("data_utils", CORE_DIR / "data_utils.py")
_sdf_operations = _load_module("sdf_operations", CORE_DIR / "sdf_operations.py")

generate_bracing_static = _bracing_generator.generate_bracing_static
infer_grid_from_scalar_fields = _data_utils.infer_grid_from_scalar_fields
meta_data_info = _data_utils.meta_data_info
stack_scalar_fields = _data_utils.stack_scalar_fields
compute_sf_operation = _sdf_operations.compute_sf_operation

app = FastAPI(title="Narnia WebUI Backend", version="0.2.0")

CACHE_ROOT = Path("output/webui_cache")


@dataclass
class DatasetState:
    dataset_id: str
    profile_fields: np.ndarray
    bracing_fields: np.ndarray
    result_fields: np.ndarray
    iso_profile: float
    iso_bracing: float
    bounds_min: Optional[list[float]]
    bounds_max: Optional[list[float]]


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, DatasetState] = {}

    def save(self, state: DatasetState) -> None:
        self._memory[state.dataset_id] = state
        npz_path = self.root / f"{state.dataset_id}.npz"
        meta_path = self.root / f"{state.dataset_id}.json"
        np.savez(
            npz_path,
            profile_fields=state.profile_fields,
            bracing_fields=state.bracing_fields,
            result_fields=state.result_fields,
            iso_profile=np.array(state.iso_profile),
            iso_bracing=np.array(state.iso_bracing),
            bounds_min=np.array(state.bounds_min if state.bounds_min is not None else []),
            bounds_max=np.array(state.bounds_max if state.bounds_max is not None else []),
        )
        meta_path.write_text(
            json.dumps(
                {
                    "dataset_id": state.dataset_id,
                    "bounds_min": state.bounds_min,
                    "bounds_max": state.bounds_max,
                },
                indent=2,
            )
        )

    def get(self, dataset_id: str) -> Optional[DatasetState]:
        if dataset_id in self._memory:
            return self._memory[dataset_id]
        npz_path = self.root / f"{dataset_id}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path, allow_pickle=False)
        bounds_min = data.get("bounds_min")
        bounds_max = data.get("bounds_max")
        state = DatasetState(
            dataset_id=dataset_id,
            profile_fields=data["profile_fields"],
            bracing_fields=data["bracing_fields"],
            result_fields=data["result_fields"],
            iso_profile=float(data["iso_profile"]),
            iso_bracing=float(data["iso_bracing"]),
            bounds_min=bounds_min.tolist() if bounds_min.size else None,
            bounds_max=bounds_max.tolist() if bounds_max.size else None,
        )
        self._memory[dataset_id] = state
        return state


cache_store = CacheStore(CACHE_ROOT)


class StaticBracingParams(BaseModel):
    num_centroids: int = Field(5, ge=1, le=6)
    ridge_sigma: float = Field(0.0, ge=0.0, le=15.0)


class StaticBracingRequest(BaseModel):
    profile_json_path: str
    bracing_json_path: Optional[str] = None
    generate_bracing: bool = True
    params: StaticBracingParams


class BooleanOperationRequest(BaseModel):
    mode: str = Field("difference", pattern="^(difference|union|intersection)$")
    profile_offset: float = 0.0
    bracing_offset: float = 0.0
    result_iso: float = 0.0
    dataset_id: Optional[str] = None


class PostprocessRequest(BaseModel):
    enable: bool = False
    close_radius: float = 0.0
    min_area: float = 0.0
    temporal_window: int = 1
    dataset_id: Optional[str] = None


class PreviewRequest(BaseModel):
    dataset_id: str
    slice_index: int = 0
    iso_threshold: float = 0.0


class MeshGenerateRequest(BaseModel):
    dataset_id: str
    source: str = Field("compute", pattern="^(compute|viewer|custom)$")
    include_result: bool = False
    include_profile: bool = True
    include_bracing: bool = True
    total_height: float = 10.0
    z_interpolation: int = 2
    iso_result: float = 0.0
    iso_profile: float = 0.0
    iso_bracing: float = 0.0
    smoothing_method: str = Field("laplacian", pattern="^(none|laplacian|taubin|laplacian_taubin)$")
    smoothing_iterations: int = 2


class ExportRequest(BaseModel):
    dataset_id: str
    output_dir: str
    export_profile: bool = False
    export_bracing: bool = False


class MeshExportRequest(BaseModel):
    mesh_id: str
    export_location: str


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/compute/static-bracing")
def compute_static_bracing(payload: StaticBracingRequest) -> dict:
    profile_path = Path(payload.profile_json_path)
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile JSON not found: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as handle:
        profile_data = json.load(handle)

    iso_profile, _, bounds_max, bounds_min = meta_data_info(profile_data)
    profile_fields, _ = stack_scalar_fields(profile_data)
    _, nx, ny = infer_grid_from_scalar_fields(profile_fields)

    if payload.generate_bracing:
        sigma_val = payload.params.ridge_sigma if payload.params.ridge_sigma > 0 else None
        bracing_fields = generate_bracing_static(
            profile_fields,
            iso_profile or 0.0,
            nx,
            ny,
            payload.params.num_centroids,
            sigma=sigma_val,
        )
        iso_bracing = 0.0
    else:
        if payload.bracing_json_path is None:
            raise HTTPException(status_code=400, detail="bracing_json_path is required when generate_bracing is false.")
        bracing_path = Path(payload.bracing_json_path)
        if not bracing_path.exists():
            raise HTTPException(status_code=404, detail=f"Bracing JSON not found: {bracing_path}")
        with bracing_path.open("r", encoding="utf-8") as handle:
            bracing_data = json.load(handle)
        iso_bracing, _, _, _ = meta_data_info(bracing_data)
        bracing_fields, _ = stack_scalar_fields(bracing_data)

    if profile_fields.shape != bracing_fields.shape:
        raise HTTPException(status_code=400, detail="Profile and bracing fields must have the same shape.")

    result_fields = compute_sf_operation(
        profile_fields,
        bracing_fields,
        iso_level_A=iso_profile or 0.0,
        iso_level_B=iso_bracing,
        mode="difference",
    )

    dataset_id = str(uuid4())
    state = DatasetState(
        dataset_id=dataset_id,
        profile_fields=profile_fields,
        bracing_fields=bracing_fields,
        result_fields=result_fields,
        iso_profile=iso_profile or 0.0,
        iso_bracing=iso_bracing,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    cache_store.save(state)

    return {
        "dataset_id": dataset_id,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "iso_profile": iso_profile or 0.0,
        "iso_bracing": iso_bracing,
        "message": "Static bracing compute complete.",
    }


@app.post("/compute/boolean")
def compute_boolean(payload: BooleanOperationRequest) -> dict:
    return {
        "message": "Boolean operation queued (MVP placeholder).",
        "payload": payload.model_dump(),
    }


@app.post("/compute/postprocess")
def compute_postprocess(payload: PostprocessRequest) -> dict:
    return {
        "message": "Postprocess queued (MVP placeholder).",
        "payload": payload.model_dump(),
    }


@app.get("/result/preview")
def get_preview(dataset_id: str, slice_index: int = 0, iso_threshold: float = 0.0) -> dict:
    state = cache_store.get(dataset_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return {
        "dataset_id": dataset_id,
        "slice_index": slice_index,
        "iso_threshold": iso_threshold,
        "slice_count": int(state.result_fields.shape[0]),
        "message": "Preview data not implemented (MVP placeholder).",
    }


@app.post("/mesh/generate")
def generate_mesh(payload: MeshGenerateRequest) -> dict:
    if cache_store.get(payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {payload.dataset_id}")
    mesh_id = str(uuid4())
    return {
        "mesh_id": mesh_id,
        "message": "Mesh generation queued (MVP placeholder).",
        "payload": payload.model_dump(),
    }


@app.get("/mesh/{mesh_id}")
def get_mesh(mesh_id: str) -> dict:
    return {
        "mesh_id": mesh_id,
        "message": "Mesh download not implemented (MVP placeholder).",
    }


@app.post("/export/npz")
def export_npz(payload: ExportRequest) -> dict:
    if cache_store.get(payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {payload.dataset_id}")
    return {
        "message": "NPZ export queued (MVP placeholder).",
        "payload": payload.model_dump(),
    }


@app.post("/mesh/export")
def export_mesh(payload: MeshExportRequest) -> dict:
    return {
        "message": "Mesh export queued (MVP placeholder).",
        "payload": payload.model_dump(),
    }
