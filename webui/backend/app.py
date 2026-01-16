from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Narnia WebUI Backend", version="0.1.0")


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
    dataset_id = str(uuid4())
    return {
        "dataset_id": dataset_id,
        "message": "Static bracing compute queued (MVP placeholder).",
        "payload": payload.model_dump(),
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
    return {
        "dataset_id": dataset_id,
        "slice_index": slice_index,
        "iso_threshold": iso_threshold,
        "message": "Preview data not implemented (MVP placeholder).",
    }


@app.post("/mesh/generate")
def generate_mesh(payload: MeshGenerateRequest) -> dict:
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
