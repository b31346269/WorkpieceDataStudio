from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .exporter import export_yolov8
from .jobs import GenerationJobs
from .schemas import (
    CandidateUpdate,
    ExportRequest,
    GenerationRequest,
    ProjectCreate,
)
from .storage import REQUIRED_QUALITY_CHECKS, ROOT, ProjectStore


app = FastAPI(title="Workpiece Data Studio", version="0.1.0")
store = ProjectStore()
jobs = GenerationJobs(store)
static_root = ROOT / "static"
app.mount("/static", StaticFiles(directory=static_root), name="static")


def api_error(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_root / "index.html")


@app.get("/api/system")
def system_status() -> dict:
    torch_available = importlib.util.find_spec("torch") is not None
    diffusers_available = importlib.util.find_spec("diffusers") is not None
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    cuda = False
    gpu = None
    vram_gb = None
    if torch_available:
        try:
            import torch

            cuda = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda else None
            vram_gb = (
                round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                if cuda
                else None
            )
        except Exception:
            pass
    return {
        "core_ready": True,
        "ml_ready": torch_available and diffusers_available,
        "yolo_ready": ultralytics_available,
        "cuda": cuda,
        "gpu": gpu,
        "vram_gb": vram_gb,
    }


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return store.list_projects()


@app.post("/api/projects")
def create_project(request: ProjectCreate) -> dict:
    try:
        return store.create_project(request.name, request.classes)
    except Exception as exc:
        raise api_error(exc) from exc


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    try:
        project = store.delete_project(project_id)
        return {"deleted": True, "project": project}
    except FileNotFoundError as exc:
        raise api_error(exc, 404) from exc
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        project = store.get_project(project_id)
        project["references"] = store.list_references(project_id)
        project["models"] = store.list_yolo_models(project_id)
        candidates = store.list_candidates(project_id)
        project["candidate_counts"] = {
            status: sum(item["status"] == status for item in candidates)
            for status in ("pending", "approved", "rejected")
        }
        return project
    except FileNotFoundError as exc:
        raise api_error(exc, 404) from exc
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/projects/{project_id}/references")
def upload_reference(
    project_id: str,
    image: UploadFile = File(...),
) -> dict:
    try:
        return store.save_reference(project_id, image.filename or "image.jpg", image.file)
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/projects/{project_id}/models")
def upload_model(
    project_id: str,
    model: UploadFile = File(...),
) -> dict:
    try:
        return store.save_yolo_model(project_id, model.filename or "model.pt", model.file)
    except Exception as exc:
        raise api_error(exc) from exc


@app.delete("/api/projects/{project_id}/models/{model_id}")
def delete_model(project_id: str, model_id: str) -> dict:
    try:
        model = store.delete_yolo_model(project_id, model_id)
        return {"deleted": True, "model": model}
    except FileNotFoundError as exc:
        raise api_error(exc, 404) from exc
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/projects/{project_id}/dataset")
def upload_dataset(
    project_id: str,
    dataset: UploadFile = File(...),
) -> dict:
    if Path(dataset.filename or "").suffix.lower() != ".zip":
        raise api_error(ValueError("Dataset must be a Roboflow YOLOv8 ZIP."))
    try:
        return store.import_yolo_zip(
            project_id,
            dataset.filename or "dataset.zip",
            dataset.file,
        )
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/api/projects/{project_id}/references/{reference_id}/image")
def reference_image(project_id: str, reference_id: str) -> FileResponse:
    try:
        _, path = store.get_reference(project_id, reference_id)
        return FileResponse(path)
    except Exception as exc:
        raise api_error(exc, 404) from exc


@app.post("/api/projects/{project_id}/generate")
def generate(project_id: str, request: GenerationRequest) -> dict:
    try:
        return jobs.submit(project_id, request)
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return jobs.get(job_id)
    except KeyError as exc:
        raise api_error(ValueError("Job not found."), 404) from exc


@app.get("/api/projects/{project_id}/candidates")
def list_candidates(project_id: str) -> list[dict]:
    try:
        return store.list_candidates(project_id)
    except Exception as exc:
        raise api_error(exc) from exc


@app.get("/api/projects/{project_id}/candidates/{candidate_id}/image")
def candidate_image(project_id: str, candidate_id: str) -> FileResponse:
    try:
        return FileResponse(store.candidate_image(project_id, candidate_id))
    except Exception as exc:
        raise api_error(exc, 404) from exc


@app.put("/api/projects/{project_id}/candidates/{candidate_id}")
def update_candidate(
    project_id: str,
    candidate_id: str,
    request: CandidateUpdate,
) -> dict:
    try:
        project = store.get_project(project_id)
        for box in request.boxes:
            if box.class_id >= len(project["classes"]):
                raise ValueError(f"Invalid class id: {box.class_id}")
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                raise ValueError(f"Invalid box: {box.id}")
        if request.status == "approved":
            missing = [
                name
                for name in REQUIRED_QUALITY_CHECKS
                if not request.quality_checks.get(name, False)
            ]
            if missing:
                raise ValueError(
                    "Complete all four mechanical quality checks before approval."
                )
        return store.update_candidate(
            project_id,
            candidate_id,
            request.status,
            [box.model_dump() for box in request.boxes],
            request.quality_checks,
        )
    except Exception as exc:
        raise api_error(exc) from exc


@app.post("/api/projects/{project_id}/export")
def export(project_id: str, request: ExportRequest) -> FileResponse:
    try:
        path, _ = export_yolov8(store, project_id, request.split_name)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=path.name,
        )
    except Exception as exc:
        raise api_error(exc) from exc
